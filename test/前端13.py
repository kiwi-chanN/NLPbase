import streamlit as st
import pandas as pd
import numpy as np
from numpy.linalg import norm
import re
from nltk.util import ngrams
from sentence_transformers import SentenceTransformer
from InstructorEmbedding import INSTRUCTOR

# 加载预训练模型
model = INSTRUCTOR('moka-ai/m3e-base')

def generate_ngrams(row, n):
    
    res= []
    
    if pd.isnull(row['评论内容']) == False:
    
        ngramss = ngrams( re.split('，|。|！|？', row['评论内容']), n)

        for grams in ngramss:
            if grams != '':
                tmp = ''
                for k in range(0, n):
                    tmp = tmp + grams[k] + '，'
                tmp = tmp[:-1]
                res.append(tmp)
        
    return res
            

# 定义一个余弦相似度，用于计算两个语句之间的相似度
def cosine_similarty(a,b):
    res = np.dot(a,b)/(norm(a)*norm(b))
    return res 


# 定义一个函数，根据每行的预测结果生成相应的回复， 本来KN这里想判断一下的，如果差评到一定地步才用针对性回复，但是我们给暴力解开了，没使用分数，直接给每个分类赋予了一个回答
def match(row):
    # 初始化空字符串，用于存储负面评论
    sentence_neg = ''
    # 初始化一个空列表，用于存储评分
    score_list = []
    res = ''
    # 遍历每个预测结果
    for k in range(0, len(row['prediction_result'])):
        # 获取当前预测结果的分类
        cat = row['prediction_result'][k]
        # 根据分类从 sentiment_specific 数据框中过滤对应的行
        sentiment_df = sentiment_specific[sentiment_specific['分类结果'] == cat]

        # 如果过滤后的 DataFrame 只有一行
        if len(sentiment_df) == 1:
            # 如果该行的 '评论' 列不为空
            if pd.isnull(sentiment_df['temp'].iloc[0]) == False:
                # 将该行的 '评论' 添加到 sentence_neg 中
                sentence_neg = sentence_neg + sentiment_df['temp'].iloc[0]
                # 评分为 -2，表示有针对性回复
                score = -2
                # 将评分添加到 score_list 中
                score_list.append(score)
            else:
                # 根据 '回复类型' 进行评分
                if sentiment_df['回复类型'].iloc[0] == '通用好评':
                    score = 1
                elif sentiment_df['回复类型'].iloc[0] == '通用中性':
                    score = 2
                elif sentiment_df['回复类型'].iloc[0] == '通用差评':
                    score = -1
                
                # 将评分添加到 score_list 中
                score_list.append(score)
    
    # 如果 score_list 不为空
    if len(score_list) > 0:
        # 根据 score_list 中的最小评分生成对应的回复
        if min(score_list) == -2:
            res = sentence_neg
        elif min(score_list) == -1:
            res = sentiment_common['差评'].iloc[0]
        elif min(score_list) == 1:
            res = (sentiment_common['好评'].iloc[0] +
                   sentiment_common['促销'].iloc[0] +
                   sentiment_common['结语'].iloc[0])
        elif min(score_list) == 2:
            res = (sentiment_common['中性评'].iloc[0] +
                   sentiment_common['推广'].iloc[0] +
                   sentiment_common['结语'].iloc[0])
                
    return res

# 假设您已经有了FAQ数据和sentiment数据

sentiment_common = pd.read_excel('~/Desktop/group1_final/商家回复1.xlsx', sheet_name='通用性回复')  # 请替换为您的文件路径
sentiment_specific = pd.read_excel('~/Desktop/group1_final/商家回复1.xlsx', sheet_name='针对性回复（new）')  # 同上


# Streamlit应用程序的主函数
def main():
    st.title("评论回复生成器")
    
    # 用户输入评论内容
    comment = st.text_area("请输入您的评论内容：")
    
    # 用户提交评论
    if st.button("提交评论"):
        # 创建一个DataFrame来存储用户输入的评论
        df = pd.DataFrame([[comment, 1]], columns=['评论内容', 'review_no'])
        
        # 生成n-gram
        df['评论内容_new'] = df.apply(generate_ngrams, axis=1, n=1)
        df = df.explode('评论内容_new')
        # 计算评论的嵌入向量
        df['query_embedding'] = df['评论内容_new'].apply(lambda x: model.encode([x]))
        query_df = df

# 初始化一个DataFrame来存储结果
        ans_df = pd.DataFrame()

        # 对 1000 条评论，循环执行以下内容
        for m in range(0, len(query_df)):

    # 如果评论内容非空，其实这个也没啥用
            if query_df['评论内容_new'].iloc[m] != '':
        
        ### huggingface
        #### 获取相似度分数

        # 获取第 m 个评论的向量
                q = query_df['query_embedding'].iloc[m]
        
        #### 在 FAQ 数据中循环计算相似度
        # df 是 FAQ 数据，所以这里获取 FAQ 向量并计算与评论向量的余弦相似度
                df = pd.read_pickle('~/Desktop/group1_final/faq_df_final.pkl')  # 请替换为您的FAQ数据文件路径
                df["similarities"] = df['faq_embedding'].apply(lambda x: cosine_similarty(x[0], q[0]))

        # 对相似度进行排序
                cnt_df = df.sort_values("similarities", ascending=False)

        ##### 设置前 5 个相似的计数
                cntcnt_df = cnt_df.head(5)
                min_score = cntcnt_df['similarities'].iloc[4]  # 获取前 5 个相似中最小的相似度
                max_score = cntcnt_df['similarities'].iloc[0]  # 获取前 5 个相似中最大的相似度

        # 对分类结果进行分组并计数
                group_df = cntcnt_df.groupby('分类结果')['faq_new'].count().reset_index(name='cnt')
                group_df = group_df.sort_values("cnt", ascending=False)
                res = group_df.head(1)['分类结果'].iloc[0]  # 获取出现次数最多的分类结果
                ratio_score = group_df.head(1)['cnt'].iloc[0] / 5  # 计算该分类结果占前 5 个相似 FAQ 的比例

        # 将结果存入 ans_df
                ans_df.at[m, 'review_no'] = query_df['review_no'].iloc[m]
                ans_df.at[m, 'query_long'] = query_df['评论内容'].iloc[m]
                ans_df.at[m, 'ngram'] = 1
                ans_df.at[m, 'query'] = query_df['评论内容_new'].iloc[m]
                ans_df.at[m, 'predict'] = res    
                ans_df.at[m, 'minscore'] = min_score  
                ans_df.at[m, 'maxscore'] = max_score          
                ans_df.at[m, 'ratio'] = ratio_score
        # 对结果进行排序，按照 review_no、query_long 和 ngram 排序
        ans_df = ans_df.sort_values(by=['review_no','query_long','ngram'], ascending = [True, False, True])
        filtered_df = ans_df[(ans_df['minscore'] >= 0.76) & (ans_df['maxscore'] >= 0.88) & (ans_df['ratio'] >= 0.7)]
        filtered_df['rank'] = filtered_df.sort_values(by=['review_no','query_long','predict','ratio','maxscore'],
                                               ascending = [True, False, False, True, True]).groupby(['review_no','query_long','predict']).cumcount(ascending=False) + 1
        filtered_df = filtered_df[filtered_df['rank']==1]

        # 对 filtered_df 按照 'predict' 进行分组，并统计每个 'predict' 分类结果中的唯一 'review_no'（评论序号）的数量
        result_df = filtered_df.groupby('predict')['review_no'].nunique().reset_index(name='review_cnt')

# 按照 'review_cnt'（评论数量）降序排序
        result_df = result_df.sort_values(['review_cnt'], ascending=False)
        sentiment_specific['temp'] = sentiment_specific['preface'] +sentiment_specific['正文']
        each_df = filtered_df.groupby('review_no')['predict'].unique().reset_index(name='prediction_result')
        
        # 对 each_df 应用 match 函数，生成对应的回复并存储在 'response' 列中
        each_df['response'] = each_df.apply(match, axis=1)
        each_df['response'] = '尊敬的宾客，感谢您选择我们四季酒店，并分享您的入住体验。'+ each_df.apply(match, axis=1) + '期待未来能有机会为您提供更满意的服务，欢迎您再次光临。'

        # 显示回复
        if not ans_df.empty:
            x = []
            for i in filtered_df['predict']:
                 x.append(i)
            st.write("经分析，识别到以下关键概念：" + str(x))
            st.write("经分析，这条评论对应的回复应为:"+ each_df['response'])

# 运行Streamlit应用程序
if __name__ == "__main__":
    main()
