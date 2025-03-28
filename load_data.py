
import pandas as pd
from sklearn.manifold import TSNE
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import tqdm
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_score, recall_score


# pd.set_option('future.no_silent_downcasting', True)

def load_data_dis(protein_name=None, seed=2, save_name=False):
    # Load the data
    data = pd.read_csv('data/MTC_combined389samples_discovry_9380prots_20240424.csv', index_col=0)
    data = data.dropna(axis=1, how='all')
    data = data.transpose()
    
    if protein_name is None:
        protein_name = data.columns.to_list()
    else:
        data = data[protein_name]
    
    label = pd.read_excel('data/MTC_total_sample_info_20240515.xlsx')
    label = label.set_index('Patient_ID')
    usefull_mask = ((label['structure_recurrence'] == 1) + (label['structure_recurrence'] == 0))>0
    
    label_clean = label.loc[usefull_mask]
    index_for_data_label = label_clean['MS_File_name'].to_list()
        
    data_clean = data.loc[index_for_data_label]    
    
    numpy_data_raw = data_clean.to_numpy()
    label_raw = label_clean['structure_recurrence'].to_numpy()
    
    
    
    # # 筛选高变量特征
    # selector = VarianceThreshold(threshold=Var_threshold)
    # numpy_data_raw = selector.fit_transform(numpy_data_raw)
    
    # print('The shape of the original data is: ', numpy_data_raw.shape)
    # print('The shape of the numpy_data_raw data is: ', numpy_data_raw.shape)
    
    # shuffle the data
    np.random.seed(seed)
    # import pdb; pdb.set_trace()
    
    
    shuffle_index = np.random.permutation(numpy_data_raw.shape[0])
    numpy_data_raw = numpy_data_raw[shuffle_index]
    label_raw = label_raw[shuffle_index]
    
    sample_list = data_clean.index.to_list()
    sample_list = [sample_list[i] for i in shuffle_index]
    
    if save_name: # with pandas
        # load the data with shuffle_index
        data_clean = data_clean.iloc[shuffle_index]
        data_clean.to_csv('data_dis.csv')
        
        
        
        
    
    return sample_list, numpy_data_raw.astype(np.float64), label_raw.astype(np.int64), protein_name


def load_data_t1(protein_name, seed=2, save_name=False):
    
    data_t1 = pd.read_csv('data/MTC_test1_102samples_10298prot.csv', index_col=0).T
    # data_t2 = pd.read_csv('data/MTC_test2_106samples_10298prot.csv', index_col=0).T
    # import pdb; pdb.set_trace()
    data_t1 = data_t1[protein_name]
    label = pd.read_excel('data/test1_shi_sample_info.xlsx', index_col=0)
    
    index_for_data_label = label.index.to_list()
    data_clean = data_t1.loc[index_for_data_label]    
    
    numpy_data_raw = data_clean.to_numpy()
    # label_raw = label['BcR/BcPD'].replace({'Yes': 1, 'No': 0}).to_numpy()
    label_raw = label['SR/SPD'].replace({'Yes': 1, 'No': 0}).to_numpy()

    np.random.seed(seed)
    shuffle_index = np.random.permutation(numpy_data_raw.shape[0])
    numpy_data_raw = numpy_data_raw[shuffle_index]
    label_raw = label_raw[shuffle_index]
    
    if save_name:
        # load the data with shuffle_index
        data_clean = data_clean.iloc[shuffle_index]
        data_clean.to_csv('data_t1.csv')
    sample_list = data_clean.index.to_list()

    # print(numpy_data_raw.shape, label_raw.shape)
    return sample_list, numpy_data_raw.astype(np.float64), label_raw.astype(np.int64)


def load_data_t2(protein_name, seed=2, save_name=False):
    
    data_t2 = pd.read_csv('data/MTC_test2_106samples_10298prot.csv', index_col=0).T
    data_t2 = data_t2[protein_name]
    label = pd.read_excel('data/MTC_total_sample_info_20240515_test2.xlsx')# .set_index('MS_File_name')

    usefull_mask = ((label['structure_recurrence'] == 1) + (label['structure_recurrence'] == 0))>0
    
    label_clean = label.loc[usefull_mask]
    index_for_data_label = label_clean['MS_File_name'].to_list()

    data_clean = data_t2.loc[index_for_data_label]    
    
    numpy_data_raw = data_clean.to_numpy()
    label_raw = label_clean['structure_recurrence'].to_numpy()

    np.random.seed(seed)
    shuffle_index = np.random.permutation(numpy_data_raw.shape[0])
    numpy_data_raw = numpy_data_raw[shuffle_index]
    label_raw = label_raw[shuffle_index]

    if save_name:
        # load the data with shuffle_index
        data_clean = data_clean.iloc[shuffle_index]
        data_clean.to_csv('data_t2.csv')
    sample_list = data_clean.index.to_list()


    # print(numpy_data_raw.shape, label_raw.shape)
    return sample_list, numpy_data_raw.astype(np.float64), label_raw.astype(np.int64)    


def visulize_data(numpy_data_raw, label_raw):

    # 初始化 t-SNE
    tsne = TSNE(n_components=2, random_state=0)

    # 对数据进行 t-SNE 降维
    data_tsne = tsne.fit_transform(numpy_data_raw)

    df_tsne = pd.DataFrame(data_tsne, columns=['Dim1', 'Dim2'])
    df_tsne['Label'] = label_raw

    # 绘制 t-SNE 结果
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Dim1', y='Dim2', hue='Label', palette='viridis', data=df_tsne, legend='full')
    plt.title('t-SNE Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend(title='Label')
    plt.show()


def train_model_return_best_model(data, label, n_estimators=50, random_state=0):
    # 5 fold cross validation
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    model_best = []
    best_auc = 0
    for train_index, test_index in kf.split(data):
        data_item = data[train_index]
        label_item = label[train_index]
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=10)
        model.fit(data_item, label_item)
        scores = roc_auc_score(label[test_index], model.predict(data[test_index]))

        if scores > best_auc:
            model_best = model
            best_auc = scores
    
    return model_best, best_auc

def test_model(data, label, model):
    predict = model.predict_proba(data)[:, 1]
    acc = np.mean(predict == label)
    auc = roc_auc_score(label, predict)
    return acc, auc

def test_model_plot(data, label, model, text='dis', save_result=False, sample_name = None):
    # predict = model.predict(data)
    predict = model.predict_proba(data)[:, 1]
    if save_result: # with pandas
        df = pd.DataFrame({'predict': predict, 'label': label})
        df.to_csv(f'{text}.csv')
    
    # import pdb; pdb.set_trace()
    # acc = np.mean(predict == label)
    fpr, tpr, thresholds = roc_curve(label, predict)
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    acc = np.mean((predict > best_threshold) == label)
    predict_label = predict > best_threshold
    precision = precision_score(label, predict > best_threshold)
    recall = recall_score(label, predict > best_threshold)
    
    roc_auc = roc_auc_score(label, predict)
    
    if sample_name is not None:
        for i in range(len(sample_name)):
            print( 'sample_name', sample_name[i], 'predict', predict[i], 'predict label', predict_label[i], 'label', label[i])
    
    # roc_auc = auc(fpr, tpr)
    
    # plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (area = {roc_auc:0.2f}),\n'
             f'acc = {acc:0.2f},\n'
             f'precision = {precision:0.2f},\n'
             f'recall = {recall:0.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{text} Dataset (ROC) Curve')
    plt.legend(loc='lower right')
    # plt.show()
    
    return acc, roc_auc


def test_feature_list(protein_name):
    
    data_dis_sl, data_dis, label_dis, protein_name = load_data_dis(protein_name)
    data_test_t1_sl, data_test_t1, label_t1 = load_data_t1(protein_name)
    data_test_t2_sl, data_test_t2, label_t2 = load_data_t2(protein_name)
    
    data_all = np.concatenate((data_dis, data_test_t1, data_test_t2), axis=0)
    label_all = np.concatenate((label_dis, label_t1, label_t2), axis=0)
    
    model, score = train_model_return_best_model(data_dis, label_dis)
    acc_dis, auc_dis = test_model(data_dis, label_dis, model)
    acc_t1, auc_t1 = test_model(data_test_t1, label_t1, model)
    acc_t2, auc_t2 = test_model(data_test_t2, label_t2, model)
    return score, auc_t1, auc_t2

def test_feature_list_plot(protein_name):
    
    data_dis_sl, data_dis, label_dis, protein_name = load_data_dis(protein_name)
    data_test_t1_sl, data_test_t1, label_t1 = load_data_t1(protein_name)
    data_test_t2_sl, data_test_t2, label_t2 = load_data_t2(protein_name)
    
    data_all = np.concatenate((data_dis, data_test_t1, data_test_t2), axis=0)
    label_all = np.concatenate((label_dis, label_t1, label_t2), axis=0)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    model, score = train_model_return_best_model(data_dis, label_dis)
    acc_dis, auc_dis = test_model_plot(data_dis, label_dis, model, text='Disvover', sample_name=data_dis_sl)
    plt.subplot(1, 3, 2)
    acc_t1, auc_t1 = test_model_plot(data_test_t1, label_t1, model, text='Test 1', sample_name=data_test_t1_sl)
    plt.subplot(1, 3, 3)
    acc_t2, auc_t2 = test_model_plot(data_test_t2, label_t2, model, text='Test 2', sample_name=data_test_t2_sl)
    plt.show()
    return score, auc_t1, auc_t2


def test_feature_list_plot_tsne(protein_name):
    
    data_dis_sl, data_dis, label_dis, protein_name = load_data_dis(protein_name)
    data_test_t1_sl, data_test_t1, label_t1 = load_data_t1(protein_name)
    data_test_t2_sl, data_test_t2, label_t2 = load_data_t2(protein_name)
    
    data_all = np.concatenate((data_dis, data_test_t1, data_test_t2), axis=0)
    label_all = np.concatenate((label_dis, label_t1, label_t2), axis=0)
    
    # vis_2d = TSNE(n_components=2, random_state=0).fit_transform(data_all)
    vis_2d = UMAP(random_state=0).fit_transform(data_all)
    
    
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.scatter(
        vis_2d[:data_dis.shape[0], 0],
        vis_2d[:data_dis.shape[0], 1],
        c=label_dis, cmap='tab10',
        s=10,
        )
    # model, score = train_model_return_best_model(data_dis, label_dis)
    # acc_dis, auc_dis = test_model_plot(data_dis, label_dis, model, text='Disvover')
    plt.subplot(1, 3, 2)
    plt.scatter(
        vis_2d[data_dis.shape[0]:data_dis.shape[0]+data_test_t1.shape[0], 0],
        vis_2d[data_dis.shape[0]:data_dis.shape[0]+data_test_t1.shape[0], 1],
        c=label_t1, cmap='tab10',
        s=10,
        )
    # acc_t1, auc_t1 = test_model_plot(data_test_t1, label_t1, model, text='Test 1')
    plt.subplot(1, 3, 3)
    plt.scatter(
        vis_2d[data_dis.shape[0]+data_test_t1.shape[0]:, 0],
        vis_2d[data_dis.shape[0]+data_test_t1.shape[0]:, 1],
        c=label_t2, cmap='tab10',
        s=10,
        )
    # acc_t2, auc_t2 = test_model_plot(data_test_t2, label_t2, model, text='Test 2')
    plt.show()
    return 0, auc_t1, auc_t2



if __name__ == '__main__':
    
    # auc_dict = {}
    # var_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    
    # for Var_threshold in tqdm.tqdm(var_list):
    #     auc_scores = []
    #     for seed in range(10):
    #         numpy_data_raw, label_raw = load_data(Var_threshold=Var_threshold, seed=seed)
    #         # clf = SVC(kernel='linear')
    #         # clf = SVC(kernel='rbf')
    #         clf = RandomForestClassifier(n_estimators=100, random_state=0)
    #         scores = cross_val_score(clf, numpy_data_raw, label_raw, cv=5, scoring='roc_auc')
    #         auc_scores.append(np.mean(scores.mean()))
            
    #         # print('The variance threshold is: ', Var_threshold)
    #         # print('The cross validation scores are: ', scores)
    #         # print('The mean of the cross validation scores is: ', scores.mean())
    #     auc_dict[Var_threshold] = np.mean(auc_scores)
    
    # print(auc_dict)
    
    data_dis, label_dis, protein_name = load_data_dis()
    
    # protein_name = protein_name[:100]
    # protein_name = ['Q96NY7_CLIC6', 'Q9ULS5_TMCC3', 'Q6PCE3_PGM2L1', 'Q15818_NPTX1', 'P14174_MIF', 'Q96GA7_SDSL', 'Q9H0Q3_FXYD6', 'Q96CX6_LRRC58', 'Q7RTS5_OTOP3', 'O15460_P4HA2', 'Q01484_ANK2', 'P58499_FAM3B', 'O76027_ANXA9', 'Q9UBU3_GHRL', 'Q9P2G3_KLHL14', 'P48506_GCLC', 'Q14249_ENDOG', 'Q86Y38_XYLT1', 'Q9BRX8_PRXL2A', 'Q6UWY0_ARSK', 'O00533_CHL1', 'P29762_CRABP1', 'Q13214_SEMA3B', 'P16035_TIMP2', 'P09211_GSTP1', 'O43490_PROM1', 'Q7Z4Q2_HEATR3', 'Q15063_POSTN', 'P13611_VCAN', 'Q9NQX7_ITM2C', 'P01303_NPY', 'Q9H8H3_TMT1A', 'Q15113_PCOLCE', 'P09958_FURIN', 'Q13591_SEMA5A', 'Q71RC9_SMIM5', 'Q96D15_RCN3', 'Q6WRI0_IGSF10', 'P19827_ITIH1', 'O00391_QSOX1', 'O60513_B4GALT4', 'Q6NSI4_RADX', 'O43505_B4GAT1', 'Q96EU7_C1GALT1C1', 'Q9GZM7_TINAGL1', 'Q7Z3B1_NEGR1', 'P02144_MB', 'P41218_MNDA', 'P50458_LHX2', 'Q9BV36_MLPH', 'P07996_THBS1', 'O75503_CLN5', 'P16402_H1_3', 'P54750_PDE1A', 'O94886_TMEM63A', 'O95395_GCNT3', 'P43251_BTD', 'P09228_CST2', 'P05771_PRKCB', 'P04271_S100B', 'P05386_RPLP1', 'Q03692_COL10A1', 'P23297_S100A1', 'Q8IXN7_RIMKLA', 'Q2PPJ7_RALGAPA2', 'Q9NVS9_PNPO', 'P10909_CLU', 'Q9UBH6_XPR1', 'Q15797_SMAD1', 'Q9H9Q2_COPS7B', 'Q9ULH0_KIDINS220', 'Q17R89_ARHGAP44', 'Q13188_STK3', 'O75976_CPD', 'Q9NPB8_GPCPD1', 'Q15700_DLG2', 'P53778_MAPK12', 'P18433_PTPRA', 'Q00796_SORD', 'Q9UN81_L1RE1', 'Q9BYT8_NLN', 'P01210_PENK', 'Q96PB7_OLFM3', 'Q16678_CYP1B1', 'Q96AQ6_PBXIP1', 'P13674_P4HA1', 'P55268_LAMB2', 'Q68CQ7_GLT8D1', 'Q96QR8_PURB', 'Q4L180_FILIP1L', 'Q53GG5_PDLIM3', 'O95757_HSPA4L', 'Q9H0P0_NT5C3A', 'Q15759_MAPK11', 'Q96N76_UROC1', 'Q9P032_NDUFAF4', 'Q13332_PTPRS', 'O43251_RBFOX2', 'Q14118_DAG1', 'Q01064_PDE1B', 'P49327_FASN', 'Q15170_TCEAL1', 'Q96I15_SCLY', 'P55809_OXCT1', 'Q8WXI2_CNKSR2', 'Q14558_PRPSAP1', 'Q8IWQ3_BRSK2', 'Q15102_PAFAH1B3', 'Q9Y334_VWA7', 'Q9BXW7_HDHD5', 'P07202_TPO', 'Q8N3R9_PALS1', 'P51178_PLCD1', 'P21709_EPHA1', 'P33527_ABCC1', 'Q32P28_P3H1', 'Q9H223_EHD4', 'O43448_KCNAB3', 'Q9P246_STIM2', 'O43766_LIAS', 'P06213_INSR', 'O75460_ERN1', 'P21796_VDAC1', 'Q9BX97_PLVAP', 'Q9H3G5_CPVL', 'Q53EL6_PDCD4', 'Q9UBR2_CTSZ', 'Q9C004_SPRY4', 'P49795_RGS19', 'Q7L5Y9_MAEA', 'P18577_RHCE', 'P21802_FGFR2', 'A0A1B0GV03_GOLGA6L7', 'Q15349_RPS6KA2', 'Q13950_RUNX2', 'Q92597_NDRG1', 'Q96CG8_CTHRC1', 'Q07654_TFF3', 'Q14914_PTGR1', 'O00462_MANBA', 'P30990_NTS', 'Q9NR99_MXRA5', 'O95848_NUDT14', 'Q96AP7_ESAM', 'Q687X5_STEAP4', 'P32929_CTH', 'Q9Y653_ADGRG1', 'Q01538_MYT1', 'P15291_B4GALT1', 'Q96FF7_MISP3', 'P58166_INHBE', 'Q99969_RARRES2', 'Q96S90_LYSMD1', 'P42229_STAT5A', 'Q3B7J2_GFOD2', 'Q5T013_HYI', 'Q8N6Y2_LRRC17', 'Q9NVR7_TBCCD1', 'P09486_SPARC', 'P16066_NPR1', 'O60941_DTNB', 'Q9HAN9_NMNAT1', 'O00469_PLOD2', 'P30626_SRI', 'Q9UHY7_ENOPH1', 'Q96KP4_CNDP2', 'P36021_SLC16A2', 'O95833_CLIC3', 'Q2NL98_VMAC', 'Q02809_PLOD1', 'Q6YP21_KYAT3', 'P12814_ACTN1', 'P17405_SMPD1', 'O95810_CAVIN2', 'Q6ZU67_BEND4', 'Q9P0M6_MACROH2A2', 'O95278_EPM2A', 'Q15582_TGFBI', 'Q16763_UBE2S', 'Q5TCQ9_MAGI3', 'P52292_KPNA2', 'O14734_ACOT8', 'P55058_PLTP', 'Q99523_SORT1', 'P82673_MRPS35', 'P21397_MAOA', 'O60547_GMDS', 'P32322_PYCR1', 'Q9UKX5_ITGA11', 'P21291_CSRP1', 'P29373_CRABP2', 'P08571_CD14', 'Q9BX79_STRA6', 'Q16563_SYPL1', 'Q6PHW0_IYD', 'Q9Y6U3_SCIN', 'Q8N9F7_GDPD1', 'P02649_APOE', 'P22413_ENPP1', 'Q9H840_GEMIN7', 'O60911_CTSV', 'Q6UVK1_CSPG4', 'Q6NY19_KANK3', 'Q15293_RCN1', 'Q9NZA1_CLIC5', 'Q5SZL2_CEP85L', 'Q9UK76_JPT1', 'Q9BWP8_COLEC11', 'Q9Y5V3_MAGED1', 'Q15942_ZYX', 'Q8N4X5_AFAP1L2', 'Q9UJ14_GGT7', 'Q9GZZ7_GFRA4', 'Q7L7V1_DHX32', 'Q16790_CA9', 'O00515_LAD1', 'Q16568_CARTPT', 'A8MYV0_DCDC2C', 'P01036_CST4', 'P17677_GAP43', 'P40199_CEACAM6', 'Q8N729_NPW', 'O15041_SEMA3E', 'Q6Q788_APOA5', 'Q92743_HTRA1', 'Q8WU39_MZB1', 'O15204_ADAMDEC1', 'P14138_EDN3', 'Q15828_CST6', 'Q9NYQ7_CELSR3', 'P41732_TSPAN7', 'P15559_NQO1', 'Q16651_PRSS8', 'P35442_THBS2', 'Q13946_PDE7A', 'P48061_CXCL12', 'P10092_CALCB', 'A0A0J9YXX1_IGHV5_10_1', 'P15088_CPA3', 'Q8N474_SFRP1', 'Q8IVN3_MUSTN1', 'P20711_DDC', 'Q9BXJ0_C1QTNF5', 'Q96A11_GAL3ST3', 'Q96G01_BICD1', 'Q08629_SPOCK1', 'P61812_TGFB2', 'Q16849_PTPRN', 'Q96E17_RAB3C', 'P05204_HMGN2', 'Q8NES3_LFNG', 'Q9Y2T3_GDA', 'Q9NZT1_CALML5', 'P35247_SFTPD', 'Q96QE2_SLC2A13', 'Q9Y274_ST3GAL6', 'Q9NQX5_NPDC1', 'P24593_IGFBP5', 'P07492_GRP', 'Q13361_MFAP5', 'P16519_PCSK2', 'P0DTE7_AMY1B', 'Q96Q80_DERL3', 'Q8IWL1_SFTPA2', 'P17936_IGFBP3', 'Q96CM8_ACSF2', 'P09603_CSF1', 'Q8IZP9_ADGRG2', 'Q8WUA8_TSKU', 'P32942_ICAM3', 'P43007_SLC1A4', 'O14522_PTPRT', 'O00445_SYT5', 'Q14146_URB2', 'P19021_PAM', 'P39900_MMP12', 'P31327_CPS1', 'Q96DA0_ZG16B', 'Q92932_PTPRN2', 'P16112_ACAN', 'Q8N475_FSTL5', 'Q9NPC4_A4GALT', 'Q96MZ0_GDAP1L1', 'P20337_RAB3B', 'Q92185_ST8SIA1', 'Q1L5Z9_LONRF2', 'O15327_INPP4B', 'O76038_SCGN', 'P07101_TH', 'Q14005_IL16', 'P78539_SRPX', 'P08514_ITGA2B']

    # protein_name = ['Q9GZP0_PDGFD', 'Q9Y6L7_TLL2', 'Q96GA7_SDSL', 'Q9UBY9_HSPB7', 'Q9NZW5_PALS2', 'B1AK53_ESPN', 'P28827_PTPRM', 'O15460_P4HA2', 'P58499_FAM3B', 'Q9UF11_PLEKHB1', 'Q01974_ROR2', 'Q9UBU3_GHRL', 'Q9P2G3_KLHL14', 'P48506_GCLC', 'Q14249_ENDOG', 'Q9H3H9_TCEAL2', 'P08582_MELTF', 'Q6UWY0_ARSK', 'P40261_NNMT', 'P29762_CRABP1', 'P20908_COL5A1', 'Q13214_SEMA3B', 'Q02539_H1_1', 'P49326_FMO5', 'P12830_CDH1', 'P16035_TIMP2', 'Q9BVH7_ST6GALNAC5', 'Q92598_HSPH1', 'Q7Z4Q2_HEATR3', 'P08311_CTSG', 'P13611_VCAN', 'O15230_LAMA5', 'Q9NQX7_ITM2C', 'P28161_GSTM2', 'P01303_NPY', 'Q96JG8_MAGED4', 'Q6WRI0_IGSF10', 'P19827_ITIH1', 'O60513_B4GALT4', 'Q7LFX5_CHST15', 'O43505_B4GAT1', 'Q9GZM7_TINAGL1', 'Q7Z3B1_NEGR1', 'Q14703_MBTPS1', 'P00740_F9', 'P07996_THBS1', 'Q6UX72_B3GNT9', 'Q9Y2E5_MAN2B2', 'P20160_AZU1', 'P16402_H1_3', 'P54750_PDE1A', 'Q13938_CAPS', 'Q9Y644_RFNG', 'Q9BXP8_PAPPA2', 'P25391_LAMA1', 'Q9UBX5_FBLN5', 'Q93045_STMN2', 'O95395_GCNT3', 'Q12860_CNTN1', 'P43251_BTD', 'Q7Z5N4_SDK1', 'P05771_PRKCB', 'Q86XX4_FRAS1', 'O14638_ENPP3', 'Q8N135_LGI4', 'P16473_TSHR', 'Q5T9S5_CCDC18', 'P22748_CA4', 'Q8IXN7_RIMKLA', 'Q2PPJ7_RALGAPA2', 'Q9NVS9_PNPO', 'Q9UBH6_XPR1', 'Q9H9Q2_COPS7B', 'O75364_PITX3', 'O75976_CPD', 'Q8WTS6_SETD7', 'O75369_FLNB', 'Q15700_DLG2', 'P18433_PTPRA', 'Q9UN81_L1RE1', 'Q9Y6Y0_IVNS1ABP', 'P09467_FBP1', 'Q9BYT8_NLN', 'Q8WWX9_SELENOM', 'P04066_FUCA1', 'Q9BW92_TARS2', 'O60936_NOL3', 'Q96PB7_OLFM3', 'Q16678_CYP1B1', 'P13674_P4HA1', 'Q9Y646_CPQ', 'P15924_DSP', 'Q8IW45_NAXD', 'O14681_EI24', 'Q9H0P0_NT5C3A', 'P98196_ATP11A', 'Q15759_MAPK11', 'P22352_GPX3', 'Q86UX6_STK32C', 'Q96N76_UROC1', 'Q9P032_NDUFAF4', 'Q8NBT3_TMEM145', 'O43251_RBFOX2', 'P62070_RRAS2', 'P16455_MGMT', 'Q9H008_LHPP', 'Q15170_TCEAL1', 'P11169_SLC2A3', 'Q96I15_SCLY', 'P07858_CTSB', 'Q14554_PDIA5', 'Q8IWQ3_BRSK2', 'Q15102_PAFAH1B3', 'Q9Y334_VWA7', 'P34896_SHMT1', 'Q9BXW7_HDHD5', 'Q06033_ITIH3', 'P07202_TPO', 'P21709_EPHA1', 'Q9H223_EHD4', 'O43448_KCNAB3', 'Q9P246_STIM2', 'Q99962_SH3GL2', 'Q8IZ41_RASEF', 'O75460_ERN1', 'A0A0J9YX94_PNMA6F', 'Q9UBR2_CTSZ', 'Q5K4L6_SLC27A3', 'Q99715_COL12A1', 'P49795_RGS19', 'Q9BQI0_AIF1L', 'O14495_PLPP3', 'P10415_BCL2', 'Q96H79_ZC3HAV1L', 'P18577_RHCE', 'Q9UI17_DMGDH', 'Q86UN3_RTN4RL2', 'Q9H936_SLC25A22', 'Q13950_RUNX2', 'Q96AQ8_MCUR1', 'Q92597_NDRG1', 'Q96CG8_CTHRC1', 'O75882_ATRN', 'Q14914_PTGR1', 'Q15198_PDGFRL', 'O00462_MANBA', 'Q765P7_MTSS2', 'Q09328_MGAT5', 'Q14393_GAS6', 'Q6PCB6_ABHD17C', 'O95848_NUDT14', 'Q96HC4_PDLIM5', 'Q96AP7_ESAM', 'P43234_CTSO', 'P49184_DNASE1L1', 'Q13015_MLLT11', 'O75607_NPM3', 'Q6KF10_GDF6', 'Q9Y653_ADGRG1', 'Q9Y6M1_IGF2BP2', 'O14498_ISLR', 'Q96EE4_CCDC126', 'P15291_B4GALT1', 'O43529_CHST10', 'Q96FF7_MISP3', 'Q99969_RARRES2', 'Q3B7J2_GFOD2', 'Q5JRM2_CXorf66', 'Q5T013_HYI', 'Q8N6Y2_LRRC17', 'Q5T1V6_DDX59', 'Q8WX93_PALLD', 'P17342_NPR3', 'P16066_NPR1', 'O60941_DTNB', 'Q14112_NID2', 'O00469_PLOD2', 'P37235_HPCAL1', 'P47712_PLA2G4A', 'O15481_MAGEB4', 'Q6NUS6_TCTN3', 'Q9Y662_HS3ST3B1', 'P14649_MYL6B', 'Q02809_PLOD1', 'P12814_ACTN1', 'Q16827_PTPRO', 'Q96MM6_HSPA12B', 'Q9P0M6_MACROH2A2', 'Q15582_TGFBI', 'Q5TCQ9_MAGI3', 'P55058_PLTP', 'P82673_MRPS35', 'P21397_MAOA', 'Q01804_OTUD4', 'Q7L513_FCRLA', 'P32322_PYCR1', 'Q9UKX5_ITGA11', 'P21291_CSRP1', 'P08571_CD14', 'Q16563_SYPL1', 'Q6PHW0_IYD', 'Q9Y6U3_SCIN', 'P22413_ENPP1', 'P05230_FGF1', 'P52429_DGKE', 'Q6UVK1_CSPG4', 'P35219_CA8', 'Q6NY19_KANK3', 'Q9C0D9_SELENOI', 'Q15293_RCN1', 'Q5SZL2_CEP85L', 'Q9BWP8_COLEC11', 'Q15942_ZYX', 'Q8N4X5_AFAP1L2', 'P02585_TNNC2', 'Q9UJ14_GGT7', 'Q9NY59_SMPD3', 'Q92626_PXDN', 'Q9HB40_SCPEP1', 'Q96DC8_ECHDC3', 'Q9H4F8_SMOC1', 'P01036_CST4', 'P17677_GAP43', 'P07093_SERPINE2', 'O15041_SEMA3E', 'Q13103_SPP2', 'Q8WU39_MZB1', 'P07988_SFTPB', 'P41732_TSPAN7', 'Q9UBS3_DNAJB9', 'P15559_NQO1', 'Q92752_TNR', 'Q86WI1_PKHD1L1', 'Q9Y328_NSG2', 'Q9NZV8_KCND2', 'P35442_THBS2', 'P22676_CALB2', 'P22003_BMP5', 'Q13822_ENPP2', 'P05976_MYL1', 'A0A0J9YXX1_IGHV5_10_1', 'O14792_HS3ST1', 'P07196_NEFL', 'P15088_CPA3', 'Q8N474_SFRP1', 'P31151_S100A7', 'Q8IVN3_MUSTN1', 'P15090_FABP4', 'Q8WYJ6_SEPTIN1', 'P08590_MYL3', 'Q9BXJ0_C1QTNF5', 'O60218_AKR1B10', 'Q96G01_BICD1', 'Q9UKR0_KLK12', 'P01033_TIMP1', 'P06881_CALCA', 'Q16849_PTPRN', 'Q96HF1_SFRP2', 'Q96E17_RAB3C', 'P69891_HBG1', 'P69892_HBG2', 'P0C0L4_C4A', 'Q8NES3_LFNG', 'O00194_RAB27B', 'Q9Y6N6_LAMC3', 'Q9BVA1_TUBB2B', 'O15240_VGF', 'Q9NZT1_CALML5', 'Q6PUV4_CPLX2', 'P35247_SFTPD', 'Q7L5N7_LPCAT2', 'P06731_CEACAM5', 'Q96QE2_SLC2A13', 'Q9NQX5_NPDC1', 'P21579_SYT1', 'Q16352_INA', 'P24593_IGFBP5', 'Q96S96_PEBP4', 'P12883_MYH7', 'Q13361_MFAP5', 'Q96Q80_DERL3', 'Q8IWL1_SFTPA2', 'P05408_SCG5', 'Q8WUA8_TSKU', 'P32942_ICAM3', 'P01909_HLA_DQA1', 'P01037_CST1', 'Q9UBX7_KLK11', 'P43007_SLC1A4', 'Q14146_URB2', 'O60704_TPST2', 'P08493_MGP', 'P39900_MMP12', 'P31327_CPS1', 'Q96DA0_ZG16B', 'Q92932_PTPRN2', 'Q9Y240_CLEC11A', 'P12259_F5', 'Q9NPC4_A4GALT', 'Q92185_ST8SIA1', 'Q1L5Z9_LONRF2', 'O76038_SCGN', 'Q05639_EEF1A2', 'Q50LG9_LRRC24', 'Q9UBX1_CTSF', 'P08514_ITGA2B']    
    # protein_name = ['Q96AP7_ESAM', 'P58499_FAM3B', 'P22676_CALB2', 'Q71RC9_SMIM5', 'P27658_COL8A1', 'P42229_STAT5A', 'P19827_ITIH1', 'Q9H008_LHPP', 'Q9Y646_CPQ', 'Q8N9F7_GDPD1', 'Q13214_SEMA3B', 'P51159_RAB27A', 'P20908_COL5A1', 'P01909_HLA_DQA1', 'Q92752_TNR', 'Q9Y2T3_GDA', 'Q9NYX4_CALY', 'P55259_GP2', 'Q9Y6Y0_IVNS1ABP', 'Q6PUV4_CPLX2', 'Q8IZ41_RASEF', 'P30990_NTS', 'P13611_VCAN', 'Q8IZJ3_CPAMD8', 'P24557_TBXAS1', 'Q14192_FHL2', 'P15559_NQO1', 'Q9BXW7_HDHD5', 'O75608_LYPLA1', 'Q14703_MBTPS1', 'Q96KA5_CLPTM1L', 'P28289_TMOD1', 'Q08257_CRYZ', 'Q9BW92_TARS2', 'O75884_RBBP9', 'Q9UBY9_HSPB7', 'P17342_NPR3', 'Q8WXD2_SCG3', 'P16112_ACAN', 'Q1L5Z9_LONRF2', 'O95833_CLIC3', 'Q9Y274_ST3GAL6', 'P07202_TPO', 'A0A1B0GV03_GOLGA6L7', 'Q86XX4_FRAS1', 'Q7Z404_TMC4', 'Q96CG8_CTHRC1', 'Q13228_SELENBP1', 'Q9UN75_PCDHA12', 'Q7L7V1_DHX32', 'Q9UJ14_GGT7', 'Q16827_PTPRO', 'O60513_B4GALT4', 'Q5VTB9_RNF220', 'Q6WRI0_IGSF10', 'Q9BQI0_AIF1L', 'P09211_GSTP1', 'Q9H9Q2_COPS7B', 'Q6ZN30_BNC2', 'P35219_CA8', 'Q2PPJ7_RALGAPA2', 'P14174_MIF', 'Q9Y240_CLEC11A', 'Q02539_H1_1', 'Q03692_COL10A1', 'Q99574_SERPINI1', 'Q8N135_LGI4', 'Q96C36_PYCR2', 'Q09328_MGAT5']
    # protein_name = ['P58499_FAM3B', 'P22676_CALB2', 'P10092_CALCB', 'Q14554_PDIA5', 'P27658_COL8A1', 'Q8NBT3_TMEM145', 'P42229_STAT5A', 'Q8N729_NPW', 'P19827_ITIH1', 'Q9Y2E5_MAN2B2', 'Q9Y6M1_IGF2BP2', 'Q8WTS6_SETD7', 'Q8IWL1_SFTPA2', 'Q96HF1_SFRP2', 'P02144_MB', 'P05386_RPLP1', 'O75607_NPM3', 'P01210_PENK', 'Q13214_SEMA3B', 'P20908_COL5A1', 'P62070_RRAS2', 'Q8WU39_MZB1', 'O15230_LAMA5', 'O14734_ACOT8', 'P28827_PTPRM', 'P22003_BMP5', 'P82673_MRPS35', 'O14792_HS3ST1', 'P14138_EDN3', 'P08590_MYL3', 'Q9Y6Y0_IVNS1ABP', 'Q6PUV4_CPLX2', 'Q8IZ41_RASEF', 'P30990_NTS', 'P13611_VCAN', 'Q8IZJ3_CPAMD8', 'P24557_TBXAS1', 'Q14192_FHL2', 'P15559_NQO1', 'Q9BXW7_HDHD5', 'O75608_LYPLA1', 'Q96KA5_CLPTM1L', 'P28289_TMOD1', 'O14495_PLPP3', 'Q08257_CRYZ', 'Q9BW92_TARS2', 'O75884_RBBP9', 'Q9UBY9_HSPB7', 'P17342_NPR3', 'Q8WXD2_SCG3', 'P16112_ACAN', 'P07492_GRP', 'Q1L5Z9_LONRF2', 'O95833_CLIC3', 'Q9Y274_ST3GAL6', 'Q8N3R9_PALS1', 'P07202_TPO', 'A0A1B0GV03_GOLGA6L7', 'Q86XX4_FRAS1', 'Q7Z404_TMC4', 'Q96CG8_CTHRC1', 'Q13228_SELENBP1', 'Q9UN75_PCDHA12', 'Q7L7V1_DHX32', 'Q9UJ14_GGT7', 'Q16827_PTPRO', 'Q5VTB9_RNF220', 'Q6WRI0_IGSF10', 'Q96E17_RAB3C', 'P09211_GSTP1', 'P20160_AZU1', 'Q9H9Q2_COPS7B', 'O75390_CS', 'Q5K4L6_SLC27A3', 'Q9UBU3_GHRL', 'P13674_P4HA1', 'P52292_KPNA2', 'P01037_CST1', 'Q9UKX2_MYH2', 'Q9Y240_CLEC11A', 'Q02539_H1_1', 'Q15113_PCOLCE', 'P09382_LGALS1', 'Q99715_COL12A1', 'P16519_PCSK2', 'P53778_MAPK12', 'Q9UKR0_KLK12', 'P32929_CTH', 'Q9UI17_DMGDH', 'Q96FF7_MISP3', 'P31151_S100A7', 'Q12860_CNTN1', 'Q9UHY7_ENOPH1', 'Q09328_MGAT5']
    protein_name = ['P19827_ITIH1', 'Q96HF1_SFRP2', 'P01210_PENK', 'Q8N9F7_GDPD1', 'O15230_LAMA5', 'P28827_PTPRM', 'Q9C0D9_SELENOI', 'P01909_HLA_DQA1', 'Q9Y2T3_GDA', 'Q9H223_EHD4', 'Q8IX30_SCUBE3', 'P30626_SRI', 'Q3B7J2_GFOD2', 'P16455_MGMT', 'P08582_MELTF', 'Q96PB7_OLFM3', 'Q15102_PAFAH1B3', 'O00257_CBX4', 'Q9BVA1_TUBB2B', 'Q9UBY9_HSPB7', 'O15240_VGF', 'Q96QR8_PURB', 'Q8N3R9_PALS1', 'A0A1B0GV03_GOLGA6L7', 'Q7Z404_TMC4', 'Q03692_COL10A1', 'Q9NYQ7_CELSR3', 'Q99574_SERPINI1', 'Q05682_CALD1']
    
    auc_dis, auc_t1, auc_t2 = test_feature_list_plot(protein_name)
    auc_dis, auc_t1, auc_t2 = test_feature_list_plot_tsne(protein_name)

    
    print('score_dis', auc_dis)    
    print('score_t1', auc_t1)
    print('score_t2', auc_t2)

    

    