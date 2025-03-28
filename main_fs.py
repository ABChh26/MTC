# algorithm.py
import random
from deap import base, creator, tools, algorithms
from load_data import test_feature_list, load_data_dis
import pandas as pd
from datetime import datetime
import numpy as np

import argparse
import wandb 

# pd.set_option('future.no_silent_downcasting', True)

def evalOneMax(individual):
    
    protein_name_item = []
    for i in range(len(individual)):
        if individual[i] == 1:
            protein_name_item.append(protein_name[i])
    
    auc_dis, auc_t1, auc_t2 = test_feature_list(protein_name_item)
    
    print(
        'auc_dis', np.round(auc_dis, 6),
        'auc_t1', np.round(auc_t1, 6),
        'auc_t2', np.round(auc_t2, 6),
        'all', np.round(auc_dis+auc_t1+auc_t2, 6),
        "protein number:", len(protein_name_item),
        str(protein_name_item), 
        file=file_log
        )

    return auc_dis+auc_t1+auc_t2 - args.len_weight*len(protein_name_item),

def run_evolution():
    # 创建适应度和个体类
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # 注册遗传算法操作
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, len(candidate))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evalOneMax)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # 初始化种群
    population = toolbox.population(n=600)

    # 运行遗传算法
    algorithms.eaSimple(
        population, 
        toolbox, 
        cxpb=0.5, 
        mutpb=0.2, 
        ngen=400, 
        stats=None, 
        halloffame=None, 
        verbose=True
        )

    # 返回最终种群
    return population

if __name__ == '__main__':
    
    argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Process some integers.')
    
    parser.add_argument('--len_weight', type=float, default=0.1, help='weight for the length of the protein list')

    args = parser.parse_args()
    
    
    # wandb.init(project="protein_selection", entity="zangzelin", )
    
    
    # load candidate protein list
    # candidate =  pd.read_csv('data/rank3nrun150_lee_extractFeatures.csv')
    # candidate = pd.read_excel('data/DEPS_top200CV_610proteins.xlsx', )
    candidate = pd.read_csv('data/protrein_v1.txt', )
    # import pdb; pdb.set_trace()
    protein_name  = candidate['name'].tolist()
    
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    log_file_name_with_time = f'log_file_{formatted_time}.txt'
    file_log = open(log_file_name_with_time, 'w')

    # data_dis, label_dis, protein_name = load_data_dis()
    print(len(protein_name))
    run_evolution()
    # wandb.finish()