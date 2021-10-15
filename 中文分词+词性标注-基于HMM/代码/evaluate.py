from models import ShortTokenizer
from models import HmmToken
from models import HmmPosTag
from tqdm import tqdm


def word_segmentation_eval(trainfile):
    #评估分词模型

    with open(trainfile, 'r', encoding='utf8') as f:
        dataset = [line.strip().split(' ') for line in f.readlines()]
    dataset = dataset[0:6000]
    input_data = [''.join(line) for line in dataset]

    # 利用 HMM 模型进行分词
    hmm_model = HmmToken.Hmm()
    #print('123')
    hmm_model.load('./data/Hmmfenci.json')
    token_res = []
    print("HMM 分词模型：")
    
    for line in tqdm(input_data):
        token_res.append(hmm_model.cut(line))  # 预测分词
    print('hmm分割完成')
    evalutate(dataset, token_res)
    with open('./data/hmmfenci_result.txt', 'w', encoding='utf8') as f:
        for i in token_res:
            f.write(' '.join(i) + '\n')

    # 利用最短路分词模型
    st_model = ShortTokenizer.ShortTokenizer(use_freq=False)
    #st_model.train(trainfile)
    #已经训练过
    st_model.train('./data/zuiduanfenci.json',trained=True)
    token_res1 = []
    print("最短路分词模型：")
    
    for line in tqdm(input_data):
        token_res1.append(st_model.Token(line))  # 预测分词
    print('最短路径分割完成')
    evalutate(dataset, token_res1)
    
    # 保存分词结果
    with open('./data/zuiduanfenci_result.txt', 'w', encoding='utf8') as f:
        for i in token_res1:
            f.write(' '.join(i) + '\n')
    return token_res,token_res1
    #返回两个分词器的分词结果
def posTag_eval(trainfile, testfile):
    """评估词性标注模型

    """
    hmm_pos = HmmPosTag.HmmPosTag()
    hmm_pos.train(trainfile)
    posTag_res = []
    with open(trainfile, 'r', encoding='utf8') as f:
        #取前两千行
        dataset = [line.strip().split(' ') for line in f.readlines()[:2000]]
    
    with open(testfile, 'r', encoding='utf8') as f:
        print("HMM 词性标注模型：")
        for line in tqdm(f.readlines()[:2000]):
            posTag_res.append(hmm_pos.predict(line.strip()))  # 预测分词        
    evalutate(dataset, posTag_res)
    #将预测的结果返回
    with open('./data/cixingbiaozhu_result.txt', 'w', encoding='utf8') as f:
        for i in posTag_res:
            f.write(' '.join(i) + '\n')
    #返回标注结果
    return posTag_res

def evalutate(groundtruth,predict):
    """计算预测结果的准确率、召回率、F1

    Args:
        predict (list): 预测结果
        groundtruth (list): 真实结果

    Returns:
        tuple(precision, recall, f1): 精确率, 召回率, f1
        [j for j in dataset[i] if j in dataset[i]]
    """
    assert len(predict) == len(groundtruth)
    tp, fp, fn = 0, 0, 0
    for i in range(len(predict)):
        right = len([j for j in predict[i] if j in groundtruth[i]])
        tp += right
        fn += len(groundtruth[i]) - right
        fp += len(predict[i]) - right
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    print("精确率:\t{:.3%}".format(precision))
    print("召回率:\t{:.3%}".format(recall))
    print("f1:\t{:.3%}".format(f1))

if __name__ == "__main__":
    # 评估分词模型
    token_res,token_res1 = word_segmentation_eval('./data/PeopleDaily_Token.txt')
    # 评估词性标注
    # 在标准分词集合上标注词性
    #带有词性标注的句子
    trainfile = './data/PeopleDaily_clean.txt'
    #删去词性标注
    testfile = './data/PeopleDaily_Token.txt'
    posTag_eval(trainfile, testfile)
    
    


