import json
import copy
import argparse
import logging


def document_metric(pre_result, gold_result):
    pre_num=0
    gold_num=0
    total_pre_true=0
    file_num=0
    macroP,macroR,macroF=0,0,0
    microP,microR,microF=0,0,0
    for pre_id in pre_result.keys():
        doc_pre=[]
        doc_gold=[]
        for ele in pre_result[pre_id]:
            if ele[2] not in doc_pre:
                doc_pre.append(ele[2])
        for ele in gold_result[pre_id]:
            if ele[2] not in doc_gold:
                doc_gold.append(ele[2])
        file_num+=1
        doc_pre_true=0        
        pre_num+=len(doc_pre)
        gold_num+=len(doc_gold)
        
        for ele in doc_pre:
            if ele in doc_gold:
                total_pre_true+=1
                doc_pre_true+=1
                              
        if len(doc_pre)==0 and len(doc_gold)==0:
            temp_macroP,temp_macroR,temp_macroF=1,1,1
        elif len(doc_pre)==0 and len(doc_gold)!=0:
            temp_macroP,temp_macroR,temp_macroF=0,0,0
        elif len(doc_pre)!=0 and len(doc_gold)==0:
            temp_macroP,temp_macroR,temp_macroF=0,0,0
        elif len(doc_pre)!=0 and len(doc_gold)!=0:
            temp_macroP=doc_pre_true/len(doc_pre)
            temp_macroR=doc_pre_true/len(doc_gold)
            if temp_macroP+temp_macroR==0:
                temp_macroF=0
            else:
                temp_macroF=2*temp_macroP*temp_macroR/(temp_macroP+temp_macroR)
                
        macroP+=temp_macroP
        macroR+=temp_macroR
        macroF+=temp_macroF
    macroP=macroP/file_num
    macroR=macroR/file_num
    if macroP+macroR!=0:
        macroF=2*macroP*macroR/(macroP+macroR)
    else:
        macroF=0
    
    if pre_num==0 and gold_num==0:
        microP,microR,microF=1,1,1
    elif pre_num==0 and gold_num!=0:
        microP,microR,microF=0,0,0
    elif pre_num!=0 and gold_num==0:
        microP,microR,microF=0,0,0
    elif pre_num!=0 and gold_num!=0:    
        microP=total_pre_true/pre_num
        microR=total_pre_true/gold_num
        if microP+microR!=0:
            microF=2*microP*microR/(microP+microR)
        else:
            microF=0
    logging.info('......document level evaluation:......')
    logging.info(f'file num: {file_num}')
    logging.info(f'{gold_num} {pre_num} {total_pre_true}')
    logging.info('miP=%.5f, miR=%.5f, miF=%.5f' %(microP,microR,microF))
    logging.info('maP=%.5f, maR=%.5f, maF=%.5f' %(macroP,macroR,macroF))
    return microF


def mention_metric_new(pre_result, gold_result):
    gold_num=0
    NER_true_num_Ps=0
    NER_true_num_Rs=0
    NER_true_num_Pr=0
    NER_true_num_Rr=0
    pre_num=0
    NEN_true_num_Ps=0
    NEN_true_num_Rs=0
    NEN_true_num_Pr=0
    NEN_true_num_Rr=0
    miss_gold = {}

    for pmid in pre_result.keys():
        
        doc_pre=copy.deepcopy(pre_result[pmid])
        doc_gold=copy.deepcopy(gold_result[pmid])
        
        gold_num+=len(doc_gold)
        pre_num+=len(doc_pre)
        
        entity_gold={}
        for ele in doc_gold:
            entity_index=ele[0]+' '+ele[1]
            if entity_index not in entity_gold.keys():
                entity_gold[entity_index]=[ele[2]]
            else:
                # logging.info(f'over!! {ele}')
                if ele[2] not in entity_gold[entity_index]:
                    entity_gold[entity_index].append(ele[2])
        entity_pre={}
        for ele in doc_pre:
            entity_index=ele[0]+' '+ele[1]
            if entity_index not in entity_pre.keys():
                entity_pre[entity_index]=[ele[2]]
            else:
                if ele[2] not in entity_pre[entity_index]:
                    entity_pre[entity_index].append(ele[2])
        
        for pre_ele in doc_pre:
            pre_index=pre_ele[0]+' '+pre_ele[1]
            if pre_index in entity_gold.keys():
                NER_true_num_Ps+=1
                if pre_ele[2] in entity_gold[pre_index]:
                    NEN_true_num_Ps+=1

        for gold_ele in doc_gold:
            gold_index=gold_ele[0]+' '+gold_ele[1]
            if gold_index in entity_pre.keys():
                NER_true_num_Rs+=1
                if gold_ele[2] in entity_pre[gold_index]:
                    NEN_true_num_Rs+=1
        
        used = [False] * len(doc_gold)
        for pre_ele in doc_pre:
            ner_flag=0
            for i, gold_ele in enumerate(doc_gold):
                if max(int(pre_ele[0]),int(gold_ele[0])) < min(int(pre_ele[1]),int(gold_ele[1])):
                    ner_flag=1
                    if pre_ele[2]==gold_ele[2]:
                        if used[i]:
                            logging.info(f'pmid: {pmid}, multiple gold matches: {gold_ele}')
                            continue
                        NEN_true_num_Pr+=1
                        used[i] = True    
                        break
            if ner_flag==1:
                NER_true_num_Pr+=1
        miss_gold[pmid] = [doc_gold[i] for i, u in enumerate(used) if not u]

        used = [False] * len(doc_pre)
        for gold_ele in doc_gold:
            ner_flag = 0
            for i, pre_ele in enumerate(doc_pre):
                if max(int(pre_ele[0]),int(gold_ele[0])) < min(int(pre_ele[1]), int(gold_ele[1])):
                    ner_flag = 1
                    if gold_ele[2]==pre_ele[2]:
                        if used[i]:
                            logging.info(f'pmid: {pmid}, multiple pre matches: {pre_ele}')
                            continue
                        NEN_true_num_Rr += 1
                        used[i] = True
                        break
            if ner_flag == 1:
                NER_true_num_Rr += 1

    if pre_num==0 and gold_num==0:
        NER_P_s,NER_R_s,NER_F_s=1,1,1
        NER_P_r,NER_R_r,NER_F_r=1,1,1
        NEN_P_s,NEN_R_s,NEN_F_s=1,1,1
        NEN_P_r,NEN_R_r,NEN_F_r=1,1,1

    elif pre_num==0 and gold_num!=0:
        NER_P_s,NER_R_s,NER_F_s=0,0,0
        NER_P_r,NER_R_r,NER_F_r=0,0,0
        NEN_P_s,NEN_R_s,NEN_F_s=0,0,0
        NEN_P_r,NEN_R_r,NEN_F_r=0,0,0
        
    elif pre_num!=0 and gold_num==0:
        NER_P_s,NER_R_s,NER_F_s=0,0,0
        NER_P_r,NER_R_r,NER_F_r=0,0,0
        NEN_P_s,NEN_R_s,NEN_F_s=0,0,0
        NEN_P_r,NEN_R_r,NEN_F_r=0,0,0
    elif pre_num!=0 and gold_num!=0:
        
        NER_P_s=NER_true_num_Ps/pre_num
        NER_P_r=NER_true_num_Pr/pre_num
        NEN_P_s=NEN_true_num_Ps/pre_num
        NEN_P_r=NEN_true_num_Pr/pre_num
        
        NER_R_s=NER_true_num_Rs/gold_num
        NER_R_r=NER_true_num_Rr/gold_num
        NEN_R_s=NEN_true_num_Rs/gold_num
        NEN_R_r=NEN_true_num_Rr/gold_num

        NER_F_s=2*NER_P_s*NER_R_s/(NER_P_s+NER_R_s) if (NER_P_s+NER_R_s) > 0 else 0
        NER_F_r=2*NER_P_r*NER_R_r/(NER_P_r+NER_R_r) if (NER_P_r+NER_R_r) > 0 else 0
        NEN_F_s=2*NEN_P_s*NEN_R_s/(NEN_P_s+NEN_R_s) if (NEN_P_s+NEN_R_s) > 0 else 0
        NEN_F_r=2*NEN_P_r*NEN_R_r/(NEN_P_r+NEN_R_r) if (NEN_P_r+NEN_R_r) > 0 else 0
        
    logging.info('......memtion level evaluation:......')
    logging.info(f'{gold_num} {pre_num}')
    logging.info('NER P_s=%.5f, R_s=%.5f, F_s=%.5f' %(NER_P_s,NER_R_s,NER_F_s))
    logging.info('NER P_r=%.5f, R_r=%.5f, F_r=%.5f' %(NER_P_r,NER_R_r,NER_F_r))  
    logging.info('NEN P_s=%.5f, R_s=%.5f, F_s=%.5f' %(NEN_P_s,NEN_R_s,NEN_F_s))
    logging.info('NEN P_r=%.5f, R_r=%.5f, F_r=%.5f' %(NEN_P_r,NEN_R_r,NEN_F_r))
    return NEN_F_r, miss_gold


def find_maxlen_entity_nest(nest_list):
    temp_result_list={}
    for i in range(0, len(nest_list)):
        hpoid=nest_list[i][-1]
        leng=len(nest_list[i][2].split())
        if hpoid not in temp_result_list.keys():
            temp_result_list[hpoid]=nest_list[i]
        else:
            if leng>len(temp_result_list[hpoid][2].split()):
                temp_result_list[hpoid]=nest_list[i]
    new_list=[]
    for hpoid in temp_result_list.keys():
        new_list.append(temp_result_list[hpoid])
    return new_list


#turn the GSC to gold HPO id, only subtree,drop the nest entity with same hpoid, remain the longest
def GSCplus_corpus_gold(goldfile_ori, goldfile, ontology_dict):
    fin=open(goldfile_ori,'r',encoding='utf-8')
    fout=open(goldfile,'w',encoding='utf-8')
    fin_alt=open(f'{ontology_dict}/alt_hpoid.json','r',encoding='utf-8')
    fin_subtree=open(f'{ontology_dict}/lable.vocab','r',encoding='utf-8')
    alt_hpoid=json.load(fin_alt)
    fin_alt.close()
    subtree_list=fin_subtree.read().strip().split('\n')
    fin_subtree.close()
    all_gold=fin.read().strip().split('\n\n')
    fin.close()

    for doc in all_gold:
        lines=doc.split('\n')
        pmid=lines[0]
        temp_result=[]
        for i in range(2,len(lines)):
            seg=lines[i].split('\t')
            if seg[3] in alt_hpoid.keys():
                hpoid=alt_hpoid[seg[3]]
                if hpoid in subtree_list:
                    seg[3]=hpoid
                    temp_result.append(seg)
            else:
                logging.info(f'pre hpo obo no this id: {lines[i]}')
        entity_list=[]
        if len(temp_result)>1:
            first_entity=temp_result[0]
            nest_list=[first_entity]
            max_eid=int(first_entity[1])
            for i in range(1,len(temp_result)):
                segs=temp_result[i]
                if int(segs[0])> max_eid:
                    if len(nest_list)==1:
                        entity_list.append(nest_list[0])
                        nest_list=[]
                        nest_list.append(segs)
                        if int(segs[1])>max_eid:
                            max_eid=int(segs[1])
                    else:
                        tem=find_maxlen_entity_nest(nest_list)#find max entity
                        entity_list.extend(tem)
                        nest_list=[]
                        nest_list.append(segs)
                        if int(segs[1])>max_eid:
                            max_eid=int(segs[1])
                else:
                    nest_list.append(segs)
                    if int(segs[1])>max_eid:
                        max_eid=int(segs[1])
            if nest_list!=[]:
                if len(nest_list)==1:
                    entity_list.append(nest_list[0])
                else:
                    tem=find_maxlen_entity_nest(nest_list)#find max entity
                    entity_list.extend(tem)
        else:
            entity_list=temp_result
        fout.write(pmid+'\n'+lines[1]+'\n')
        for ele in entity_list:
            fout.write('\t'.join(ele)+'\n')
        fout.write('\n')
    fout.close()


def GSCplus_corpus(prefile,goldfile,ontology_dict,subtree=True):
    fin_pre=open(prefile,'r',encoding='utf-8')
    fin_gold=open(goldfile,'r',encoding='utf-8')
    fin_alt=open(f'{ontology_dict}/alt_hpoid.json','r',encoding='utf-8')
    fin_subtree=open(f'{ontology_dict}/lable.vocab','r',encoding='utf-8')
    alt_hpoid=json.load(fin_alt)
    fin_alt.close()
    subtree_list=fin_subtree.read().strip().split('\n')
    fin_subtree.close()
    all_pre=fin_pre.read().strip().split('\n\n')
    all_gold=fin_gold.read().strip().split('\n\n')
    fin_gold.close()
    fin_pre.close()
    pre_result={}
    gold_result={}
    for doc_pre in all_pre:
        lines=doc_pre.split('\n')
        pmid=lines[0]
        temp_result=[]
        for i in range(2,len(lines)):
            seg=lines[i].split('\t')
            if seg[3] in alt_hpoid.keys():
                hpoid=alt_hpoid[seg[3]]
                if subtree==True:
                    if hpoid in subtree_list:
                        temp_result.append([seg[0],seg[1],hpoid])
                else:
                    temp_result.append([seg[0],seg[1],hpoid])
            else:
                logging.info(f'pre hpo obo no this id: {lines[i]} {seg[3]}')

        pre_result[pmid]=temp_result

    for doc_gold in all_gold:
        lines=doc_gold.split('\n')
        pmid=lines[0]
        temp_result=[]
        for i in range(2,len(lines)):
            seg=lines[i].split('\t')
            for id in seg[3].split('|'):
                if id in alt_hpoid.keys():
                    hpoid=alt_hpoid[id]
                    if subtree==True:
                        if hpoid in subtree_list:
                            temp_result.append([seg[0],seg[1],hpoid])
                    else:
                        temp_result.append([seg[0],seg[1],hpoid])
                else:
                    logging.info(f'gold hpo obo no this id: {lines[i]} {id}')
        gold_result[pmid]=temp_result
    doc_f=document_metric(pre_result,gold_result)
    men_f,miss_gold=mention_metric_new(pre_result,gold_result)
    return doc_f+men_f, miss_gold