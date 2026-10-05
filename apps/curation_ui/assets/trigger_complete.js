/* Tab-completion for the trigger-word fields, from the segment's own words.
 *
 * A trigger word is not free text: it is a verbatim quote from one sentence of the report, and
 * `common.validate_annotation` rejects it when it is not. So the curator retypes words that are
 * already on the screen, and every typo is a rejected suggestion — or worse, an accepted one whose
 * span no longer locates. This turns that into a letter and a run of Tabs.
 *
 *   type "h"  →  Tab  →  "hypotonia"  →  Tab  →  "hypotonia of"  →  Tab  →  "hypotonia of the"
 *
 * Two moves, one key. When the last word you typed is an unfinished prefix — from the very first
 * letter, and from an empty field — Tab finishes it; when it is already a whole word of the
 * segment, Tab appends the word that follows it *there*, so the completion tracks the phrase
 * through the sentence rather than offering a bag of words. The hint under the field says which
 * word Tab would take, and how many others there are.
 *
 * **Every completion is a verbatim slice of the segment**, never words rejoined with spaces. A
 * report pasted out of a lab system pads its columns with tabs — `Serum Ammonia<TAB><TAB>87` —
 * and a trigger assembled with single spaces is not a quote of anything. Tabbing through that line
 * puts the tabs in the field, and `views.common.snap_trigger` repairs the typed-by-hand case on
 * the way to disk.
 *
 * The rest of the keys are the ones an editor's completion popup already trains people to expect,
 * which is the whole reason to reuse them:
 *
 *   Tab          accept what the hint shows
 *   ↑ / ↓        cycle the alternatives, when a prefix or a phrase occurs more than once
 *   Enter        done here — jump to the HPO term field
 *   Esc          dismiss; Tab and Shift-Tab are ordinary focus keys again until you type
 *   Shift-Tab    focus the previous control, exactly as it always did
 *
 * Enter and Esc exist because Tab stops being an exit. A finished phrase usually still has a next
 * word to offer, so a curator who only had Tab would be walking further into the sentence every
 * time they tried to leave the field. Ctrl-Tab, the obvious third key, is not available at all:
 * Chrome switches browser tabs on it and a page cannot preventDefault its way out.
 *
 * Neither key is swallowed. Enter is left to bubble so the `debounce=True` inputs still flush their
 * value to Dash, and the focus move is deferred a tick so it happens after that; Esc only sets a
 * flag this module reads.
 *
 * Why clientside, and why the source text is read out of the DOM: the words come from the segment
 * the field is pointed at, which for the inline editors is whatever number is currently in *that
 * row's* segment box — a value the server does not have until the row is saved. The report is
 * already rendered, so the segments are already here.
 *
 * Why the hint is a fixed-position node on <body> rather than a sibling of the input: these panels
 * are re-rendered by Dash constantly, and a node inserted into a container React owns is a node
 * React may remove, reorder, or trip over. Nothing here touches the tree Dash manages, except to
 * write a value through React's own setter (see `setValue`).
 */
(function () {
  "use strict";

  /**
   * How much of a word has to be typed before Tab will finish it. One: completion is offered from
   * the very first letter, and from an empty field (which is the `prefix === ""` branch below,
   * offering the segment's next word outright). It was three, on the theory that two letters is a
   * keystroke rather than an intention — but the alternatives are on the arrows and the hint counts
   * them, so a wide first guess costs a keypress, while a threshold costs two letters on every
   * single trigger.
   */
  var MIN_PREFIX = 1;
  /** More alternatives than this and cycling is worse than typing another letter. */
  var MAX_CANDIDATES = 8;
  /** Punctuation is not part of a word: "hypotonia," completes to "hypotonia". */
  var EDGE = /^[^0-9a-zÀ-ɏ]+|[^0-9a-zÀ-ɏ]+$/gi;

  var state = new WeakMap();   // input -> {value: string, idx: number, off: bool}
  var hint = null;

  function isTrigger(el) {
    if (!el || el.tagName !== "INPUT") return false;
    return el.id === "edit-trigger" || el.id.indexOf("ann-trigger") >= 0;
  }

  function parseId(raw) {
    if (!raw || raw.charAt(0) !== "{") return null;
    try { return JSON.parse(raw); } catch (e) { return null; }
  }

  /** The text of a rendered node, without the little HPO-code tags woven into the marks. */
  function textOf(el) {
    if (!el) return "";
    var clone = el.cloneNode(true);
    var tags = clone.querySelectorAll(".mark-tag");
    for (var i = 0; i < tags.length; i++) tags[i].remove();
    return clone.textContent || "";
  }

  function segmentByIndex(idx) {
    var segs = document.querySelectorAll(".seg");
    for (var i = 0; i < segs.length; i++) {
      var id = parseId(segs[i].getAttribute("id"));
      if (id && id.type === "seg" && Number(id.idx) === idx) return segs[i];
    }
    return null;
  }

  /** The row editor's own segment box — the one whose number decides this field's word list. */
  function siblingSegInput(id) {
    var inputs = document.querySelectorAll("input[id]");
    for (var i = 0; i < inputs.length; i++) {
      var other = parseId(inputs[i].getAttribute("id"));
      if (other && other.type === "ann-seg" && other.key === id.key && other.at === id.at) {
        return inputs[i];
      }
    }
    return null;
  }

  /**
   * The words this field may complete from.
   *
   * The New-phenotype form is pointed at whichever segment is selected, and the Edit panel already
   * renders that segment's raw text — no marks, no tags, so it is the cleanest source available.
   * A row editor is pointed at its own segment number instead, which is why the two are resolved
   * differently. Both fall back to the selected segment rather than to the whole report: completing
   * from a sentence the annotation does not name would produce a trigger that cannot locate.
   */
  function sourceFor(input) {
    if (input.id === "edit-trigger") {
      return document.querySelector(".edit-seg-body") || document.querySelector(".seg-selected");
    }
    var id = parseId(input.id);
    if (id) {
      var seg = siblingSegInput(id);
      var idx = seg && seg.value !== "" ? parseInt(seg.value, 10) : NaN;
      if (!isNaN(idx)) {
        var found = segmentByIndex(idx);
        if (found) return found;
      }
    }
    return document.querySelector(".seg-selected");
  }

  function norm(word) { return String(word).toLowerCase().replace(EDGE, ""); }

  /**
   * *text* as `{raw, tok}`: the string itself, and its words with the offsets they sit at.
   *
   * The offsets are the whole point. A report pasted out of a lab system pads its columns with
   * tabs — `Serum Ammonia<TAB><TAB>87` — and a completion assembled by joining words with single
   * spaces produces a trigger that is not a quote of anything. Slicing `raw` between two offsets
   * instead makes every completion a verbatim substring of the segment, whitespace included.
   */
  function scan(text) {
    var raw = String(text);
    var tok = [];
    var re = /\S+/g;
    var m;
    while ((m = re.exec(raw)) !== null) {
      var word = m[0].replace(EDGE, "");
      if (!word) continue;
      var lead = m[0].indexOf(word);
      tok.push({ word: word, start: m.index + lead, end: m.index + lead + word.length });
    }
    return { raw: raw, tok: tok };
  }

  /**
   * What Tab could do to *value*, best first, as `{text, insert}` pairs.
   *
   * `text` is the whole new field value — built by splicing into the string the curator actually
   * typed, so their spacing and capitalisation survive a completion. `insert` is only what the hint
   * shows.
   *
   * Extensions of an already-complete word come before completions of it, because the run of Tabs
   * this exists for is a phrase being walked forward: having typed "seizure", the next Tab should
   * reach "seizure with" rather than sideways to "seizures".
   */
  function candidates(source, value) {
    var tok = source.tok;
    var open = /\s$/.test(value) || value.trim() === "";
    var typed = value.trim() ? value.trim().split(/\s+/) : [];
    var ctx = open ? typed : typed.slice(0, -1);
    var prefix = open ? "" : norm(typed[typed.length - 1]);

    var extend = [], complete = [], seen = {};
    // Every candidate is `raw[tok[from].start : tok[to].end]` — the segment's own characters, so
    // whatever separates the words there ends up in the field rather than a space standing in for
    // it. The whole value is rebuilt, not just the tail, because the words already typed matched
    // `tok[from..]` to get here: rewriting them costs nothing and fixes their spelling too.
    function push(list, from, to, insert) {
      var text = source.raw.slice(tok[from].start, tok[to].end);
      if (seen[text] || extend.length + complete.length >= MAX_CANDIDATES) return;
      seen[text] = 1;
      list.push({ text: text, insert: insert });
    }

    for (var s = 0; s + ctx.length <= tok.length; s++) {
      var ok = true;
      for (var i = 0; i < ctx.length; i++) {
        if (norm(tok[s + i].word) !== norm(ctx[i])) { ok = false; break; }
      }
      if (!ok) continue;
      var p = s + ctx.length;
      var here = p < tok.length ? tok[p].word : null;
      if (prefix === "") {
        if (here) push(extend, s, p, here);
        continue;
      }
      if (!here) continue;
      if (norm(here) === prefix) {
        // The word is already whole: the next Tab belongs to the word after it.
        if (p + 1 < tok.length) push(extend, s, p + 1, tok[p + 1].word);
      } else if (prefix.length >= MIN_PREFIX && norm(here).indexOf(prefix) === 0) {
        push(complete, s, p, here);
      }
    }
    return extend.concat(complete);
  }

  /**
   * The HPO-term control belonging to *input* — where Enter goes.
   *
   * Named rather than derived, because "the next focusable element" is the *Use selection* button
   * in both layouts, and landing there is not what anybody means by leaving the trigger field. The
   * generic walk is only the fallback for a layout that has moved on without this function.
   */
  function nextField(input) {
    if (input.id === "edit-trigger") return document.getElementById("edit-hpo");
    var id = parseId(input.id);
    if (id) {
      var nodes = document.querySelectorAll("[id]");
      for (var i = 0; i < nodes.length; i++) {
        var other = parseId(nodes[i].getAttribute("id"));
        if (other && other.type === "ann-hpo" && other.key === id.key && other.at === id.at) {
          return nodes[i];
        }
      }
    }
    return null;
  }

  function focusNext(input) {
    var field = nextField(input);
    // A dcc.Dropdown is a div carrying the id; the thing that takes focus is the input inside it.
    var target = field ? (field.querySelector("input") || field) : null;
    if (!target) {
      var order = document.querySelectorAll("input, select, textarea, button, [tabindex]");
      for (var i = 0; i < order.length; i++) {
        if (order[i] === input) { target = order[i + 1] || null; break; }
      }
    }
    if (target && target.focus) target.focus();
  }

  function pick(input) {
    var source = scan(textOf(sourceFor(input)));
    if (!source.tok.length) return null;
    var value = input.value || "";
    var found = candidates(source, value);
    if (!found.length) return null;
    var kept = state.get(input);
    if (kept && kept.off && kept.value === value) return null;
    var idx = kept && kept.value === value ? kept.idx % found.length : 0;
    state.set(input, { value: value, idx: idx, off: false });
    return { list: found, idx: idx, choice: found[idx] };
  }

  /**
   * Write *text* into the field the way a keystroke would.
   *
   * Assigning `input.value` on a React-controlled input updates the DOM and nothing else: the
   * component's own state still holds the old string and overwrites this on its next render, so the
   * completion would flicker away and never reach the server. Going through the prototype's setter
   * and firing a bubbling `input` event is what makes React — and therefore Dash — see it.
   */
  function setValue(input, text) {
    var descriptor = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value");
    if (descriptor && descriptor.set) descriptor.set.call(input, text);
    else input.value = text;
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.setSelectionRange(text.length, text.length);
  }

  function hintNode() {
    if (!hint) {
      hint = document.createElement("div");
      hint.className = "trigger-hint";
      hint.setAttribute("aria-hidden", "true");
      document.body.appendChild(hint);
    }
    return hint;
  }

  function hideHint() {
    if (hint) hint.style.display = "none";
  }

  function keysNode() {
    var keys = document.createElement("span");
    keys.className = "trigger-hint-keys";
    keys.textContent = "Enter → term · Esc dismisses";
    return keys;
  }

  function showHint(input) {
    var found = pick(input);
    if (!found) { hideHint(); return; }
    var node = hintNode();
    var text = "Tab → " + found.choice.insert;
    if (found.list.length > 1) {
      text += "   (" + (found.idx + 1) + "/" + found.list.length + " ↑↓)";
    }
    // The two exits are spelled out on the hint itself. They are the keys a curator needs exactly
    // when Tab has stopped being one, which is the moment they are least likely to go looking.
    node.textContent = text;
    node.appendChild(keysNode());
    node.style.display = "block";
    var rect = input.getBoundingClientRect();
    node.style.left = rect.left + "px";
    node.style.top = (rect.bottom + 4) + "px";
    node.style.maxWidth = Math.max(rect.width, 180) + "px";
  }

  function reposition() {
    var el = document.activeElement;
    if (hint && hint.style.display === "block" && isTrigger(el)) showHint(el);
    else hideHint();
  }

  document.addEventListener("keydown", function (event) {
    var input = event.target;
    if (!isTrigger(input)) return;

    // Leave the field. Deliberately not swallowed: the row editors are ``debounce=True``, so Enter
    // is also how their value reaches Dash, and the focus move waits a tick for that to happen.
    if (event.key === "Enter" && !event.shiftKey) {
      setTimeout(function () { focusNext(input); }, 0);
      hideHint();
      return;
    }

    // Dismiss. The field keeps what it holds; Tab and Shift-Tab go back to being focus keys until
    // the next keystroke revives the suggestion.
    if (event.key === "Escape") {
      var now = state.get(input);
      state.set(input, {
        value: input.value || "", idx: now ? now.idx : 0, off: true,
      });
      hideHint();
      return;
    }

    // Cycle. Arrows rather than Shift-Tab, which every keyboard user already owns for going back —
    // and an autocomplete's alternatives are what arrows mean everywhere else on the web.
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      var open = pick(input);
      if (!open || open.list.length < 2) return;
      var step = event.key === "ArrowDown" ? 1 : open.list.length - 1;
      event.preventDefault();
      state.set(input, {
        value: input.value || "",
        idx: (open.idx + step) % open.list.length,
        off: false,
      });
      showHint(input);
      return;
    }

    if (event.key !== "Tab" || event.shiftKey || event.ctrlKey || event.altKey || event.metaKey) {
      return;
    }

    var found = pick(input);
    // No candidate, or dismissed: Tab is Tab. Swallowing it would trap the keyboard in a field that
    // has nothing to offer, which is worse than not having completion at all.
    if (!found) return;

    event.preventDefault();
    setValue(input, found.choice.text);
    state.set(input, { value: found.choice.text, idx: 0, off: false });
    showHint(input);
  }, true);

  document.addEventListener("input", function (event) {
    // Typing revives a dismissed field: Esc means "not for this string", not "never again".
    if (isTrigger(event.target)) { state.delete(event.target); showHint(event.target); }
  });
  document.addEventListener("focusin", function (event) {
    if (isTrigger(event.target)) showHint(event.target); else hideHint();
  });
  document.addEventListener("focusout", function (event) {
    if (isTrigger(event.target)) hideHint();
  });
  window.addEventListener("scroll", reposition, true);
  window.addEventListener("resize", reposition);
})();
