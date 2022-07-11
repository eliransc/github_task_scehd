import os.path
import pickle5 as pkl
from simulator import Simulator
import pandas as pd
import argparse
import numpy as np
import sys

# curr_num = 7

# a1 = 1.5
# a2 = 0.25
# a3 = 0.75
# a4 = 0.5
# a5 = 3


class MyPlanner:

    def check_if_there_is_possible_match(self, available_resources, unassigned_tasks, resource_pool):

        for task in unassigned_tasks:
            for resource in available_resources:
                if resource in resource_pool[task.task_type]:
                    return True
        return False



    def get_task_out_prob(self, df_trans, task_):
        curr_ind = df_trans.loc[df_trans['task'] == task_, :].index[0]
        if df_trans.iloc[curr_ind, 1:].sum()>0:
            return df_trans.iloc[curr_ind, -1] / df_trans.iloc[curr_ind, 1:].sum()
        else:
            return 0

    def give_task_ranking(self, df_mean_var, avail_res, task):

        ######################################################
        ##### Begining: Ranking tasks within resource ########
        ######################################################
        if (df_mean_var.shape[0] > 0) & ('mean_'+str(task) in list(df_mean_var.columns)):
            df_res_ranking = pd.DataFrame([])
            for ind_res, res in enumerate(avail_res):
                curr_ind = df_res_ranking.shape[0]
                if df_mean_var.loc[df_mean_var['resource'] == res, 'mean_' + str(task)].shape[0] > 0:
                    if df_mean_var.loc[df_mean_var['resource'] == res, 'mean_' + str(task)].item()>0:
                        df_res_ranking.loc[curr_ind, 'resource'] = res
                        df_res_ranking.loc[curr_ind, 'task_mean'] = df_mean_var.loc[
                            df_mean_var['resource'] == res, 'mean_' + str(task)].item()

            if df_res_ranking.shape[0] > 0:
                df_res_ranking = df_res_ranking.sort_values(by='task_mean').reset_index()
                df_res_ranking['Ranking'] = np.arange(df_res_ranking.shape[0])
                return df_res_ranking


    def give_resource_ranking(self, df_mean_var, resource, unassigned_tasks_):

        ######################################################
        ##### Begining: Ranking tasks within resource ########
        ######################################################

        if (df_mean_var.shape[0] > 0) & (resource in list( df_mean_var['resource'])):  # if df_mean_var initiated and if we have knowegle about resource
            # task_names = [col for col in df_mean_var.columns if col.startswith('mean')]
            df_ranking_tasks = pd.DataFrame([])
            unassigned = [task for task in unassigned_tasks_ if 'mean_' + task in df_mean_var.columns]
            if len(unassigned) > 0:
                for ind, curr_task in enumerate(unassigned):
                    df_ranking_tasks.loc[ind, 'task_name'] = curr_task
                    df_ranking_tasks.loc[ind, 'task_mean'] = df_mean_var.loc[
                        df_mean_var['resource'] == resource, 'mean_'+curr_task].item()

                df_ranking_task = df_ranking_tasks.loc[df_ranking_tasks['task_mean'] > 0, :].sort_values(
                    by='task_mean').reset_index()
                df_ranking_task['Ranking'] = np.arange(df_ranking_task.shape[0])
            # if df_ranking_task.loc[df_ranking_task['task_name'] == task.task_type, :].shape[0] > 0:
            #     task_ranking = df_ranking_task.loc[df_ranking_task['task_name'] == task.task_type, :].index[0]
            #     print(task_ranking, df_ranking_task.shape[0])

                return df_ranking_task

        #################################################
        ##### End: Ranking tasks within resource ########
        #################################################

    def plan(self, available_resources, unassigned_tasks, resource_pool, a1,a2,a3,a4,a5, curr_num):

        path_freq_transition = './data/freq_transition_path_' + str(curr_num) + '.pkl'
        mean_path = './data/pd_mean_var_path_' + str(curr_num) + '.pkl'
        path = './data/pd_path_' + str(curr_num) + '.pkl'

        if os.path.exists(path):

            df = pkl.load(open(path, 'rb'))

            curr_df_status = df.index[-1]
        else:
            curr_df_status = 0
        # print(df.index[-1])


        if not os.path.exists(mean_path):
            df_mean_var = pd.DataFrame(columns=['resource'])
        else:
            df_mean_var = pkl.load(open(mean_path, 'rb'))

        assignments = []
        # assign the first unassigned task to the first available resource, the second task to the second resource, etc.


        unassigned_tasks_ = [task.task_type for task in unassigned_tasks]

        dict_ranking_tasks = {}
        for resource in available_resources:
            dict_ranking_tasks[resource] = self.give_resource_ranking(df_mean_var, resource, set(unassigned_tasks_))

        dict_ranking_resource = {}

        for task in set(unassigned_tasks_):
            dict_ranking_resource[task] = self.give_task_ranking(df_mean_var, available_resources, task)


        df_combs_score = pd.DataFrame([])
        # print(dict_ranking_resource)
        # print('****************Start assignment*************************')
        for task in set(unassigned_tasks_):
            for resource in available_resources:
                if resource in resource_pool[task]:

                    mean_val = -1
                    var_val = -1
                    if df_mean_var.shape[0] > 0:
                        if 'mean_' + task in df_mean_var.columns:
                            if df_mean_var.loc[df_mean_var['resource'] == resource, 'mean_' + task].shape[
                                0] > 0:
                                if df_mean_var.loc[
                                    df_mean_var['resource'] == resource, 'mean_' + task].item() > 0:
                                    mean_val = df_mean_var.loc[
                                        df_mean_var['resource'] == resource, 'mean_' + task].item()
                                    var_val = df_mean_var.loc[
                                        df_mean_var['resource'] == resource, 'var_' + task].item()

                    curr_df = dict_ranking_resource[task]
                    res_rank = -1
                    if not curr_df is None:
                        if not (curr_df == None).all()[0]:
                            if curr_df.loc[curr_df['resource'] == resource, 'Ranking'].shape[0] > 0:
                                res_rank = curr_df.loc[curr_df['resource'] == resource, 'Ranking'].item()

                    curr_df = dict_ranking_tasks[resource]
                    task_rank = -1
                    if not curr_df is None:
                        if not (curr_df == None).all()[0]:
                            if curr_df.loc[curr_df['task_name'] == task, 'Ranking'].shape[0]:
                                task_rank = curr_df.loc[curr_df['task_name'] == task, 'Ranking'].item()

                    if os.path.exists(path_freq_transition):
                        df_freq_transition = pkl.load(open(path_freq_transition, 'rb'))
                        prob = self.get_task_out_prob(df_freq_transition, task)
                        if not prob >= 0:
                            prob = -1
                    else:
                        prob = -1

                    # print(resource, task, mean_val, var_val, res_rank, task_rank, prob)
                    curr_ind = df_combs_score.shape[0]
                    df_combs_score.loc[curr_ind, 'resource'] = resource
                    df_combs_score.loc[curr_ind, 'task'] = task
                    df_combs_score.loc[curr_ind, 'mean_val'] = mean_val
                    df_combs_score.loc[curr_ind, 'var_val'] = var_val
                    df_combs_score.loc[curr_ind, 'res_rank'] = res_rank
                    df_combs_score.loc[curr_ind, 'task_rank'] = task_rank
                    df_combs_score.loc[curr_ind, 'prob'] = prob
                    df_combs_score.loc[curr_ind, 'tot_score'] = a1*mean_val*+a2*var_val+a3*res_rank+a4*task_rank-a5*prob


        # print('****************End assignment*************************')

        unassigned_tasks_ = [task.task_type for task in unassigned_tasks]

        dict_ranking_tasks = {}
        for resource in available_resources:
            dict_ranking_tasks[resource] = self.give_resource_ranking(df_mean_var, resource, set(unassigned_tasks_))

        dict_ranking_resource = {}

        for task in set(unassigned_tasks_):
            dict_ranking_resource[task] = self.give_task_ranking(df_mean_var, available_resources, task)

        # print(dict_ranking_resource)
        df_sched_score = pd.DataFrame([])
        # print('****************Start assignment*************************')
        for task in set(unassigned_tasks_):
            for resource in available_resources:
                if resource in resource_pool[task]:

                    mean_val = -1
                    var_val = -1
                    if df_mean_var.shape[0] > 0:
                        if 'mean_' + task in df_mean_var.columns:
                            if df_mean_var.loc[df_mean_var['resource'] == resource, 'mean_' + task].shape[0] > 0:
                                if df_mean_var.loc[
                                    df_mean_var['resource'] == resource, 'mean_' + task].item() > 0:
                                    mean_val = df_mean_var.loc[
                                        df_mean_var['resource'] == resource, 'mean_' + task].item()
                                    var_val = df_mean_var.loc[
                                        df_mean_var['resource'] == resource, 'var_' + task].item()

                    curr_df = dict_ranking_resource[task]
                    res_rank = -1
                    if not curr_df is None:
                        if not (curr_df == None).all()[0]:
                            if curr_df.loc[curr_df['resource'] == resource, 'Ranking'].shape[0] > 0:
                                res_rank = curr_df.loc[curr_df['resource'] == resource, 'Ranking'].item()

                    curr_df = dict_ranking_tasks[resource]
                    task_rank = -1
                    if not curr_df is None:
                        if not (curr_df == None).all()[0]:
                            if curr_df.loc[curr_df['task_name'] == task, 'Ranking'].shape[0]:
                                task_rank = curr_df.loc[curr_df['task_name'] == task, 'Ranking'].item()

                    if os.path.exists(path_freq_transition):
                        df_freq_transition = pkl.load(open(path_freq_transition, 'rb'))
                        prob = self.get_task_out_prob(df_freq_transition, task)
                        if not prob >= 0:
                            prob = -1
                    else:
                        prob = -1

                    # print(resource, task, mean_val, var_val, res_rank, task_rank, prob)
                    curr_ind = df_sched_score.shape[0]
                    df_sched_score.loc[curr_ind, 'resource'] = resource
                    df_sched_score.loc[curr_ind, 'task'] = task
                    df_sched_score.loc[curr_ind, 'mean_val'] = mean_val
                    df_sched_score.loc[curr_ind, 'var_val'] = var_val
                    df_sched_score.loc[curr_ind, 'res_rank'] = res_rank
                    df_sched_score.loc[curr_ind, 'task_rank'] = task_rank
                    df_sched_score.loc[curr_ind, 'prob'] = prob
                    df_sched_score.loc[curr_ind, 'tot_score'] = a1*mean_val+a2*var_val+a3*res_rank+a4*task_rank+a5*prob

        if (df_sched_score.shape[0] > 0 and curr_df_status > 2000):  # if there is at least one task resource combination in  df_sched_score
            df_sched_score = df_sched_score.sort_values(by='tot_score').reset_index()

            # print('****************End assignment*************************')
            while self.check_if_there_is_possible_match(available_resources, unassigned_tasks, resource_pool):
                for ind in range(df_sched_score.shape[0]):

                    task = df_sched_score.loc[ind, 'task']
                    res = df_sched_score.loc[ind, 'resource']
                    inds_tasks = [task_ind for task_ind in range(len(unassigned_tasks)) if (unassigned_tasks[task_ind].task_type==task)]
                    if len(inds_tasks) > 0:
                        if res in available_resources:
                            if res in resource_pool[task]:
                                assignments.append((unassigned_tasks[inds_tasks[0]], res))
                                # print(res, unassigned_tasks[inds_tasks[0]])
                                available_resources.remove(res)
                                unassigned_tasks.pop(inds_tasks[0])

                                break

        else:

            for task in unassigned_tasks:
                for resource in available_resources:
                    if resource in resource_pool[task.task_type]:

                        available_resources.remove(resource)
                        assignments.append((task, resource))
                        # print(task.task_type, resource)
                        break

        return assignments


    def report(self, event, curr_num):

        path = './data/pd_path_' +str(curr_num)+'.pkl'

        path_freq_transition =  './data/freq_transition_path_' +str(curr_num)+'.pkl'


        if not os.path.exists(path):
            df = pd.DataFrame([])
        else:
            df = pkl.load(open(path, 'rb'))
        curr_ind = df.shape[0]
        if curr_ind > 1000:
            df = df.iloc[1:,:]
            curr_ind = df.index[-1]+1

        df.loc[curr_ind,'case_id'] = event.case_id
        df.loc[curr_ind, 'task'] = str(event.task)
        df.loc[curr_ind, 'timestamp'] = event.timestamp
        df.loc[curr_ind, 'date_time'] = str(event).split('\t')[2]
        df.loc[curr_ind, 'resource'] = event.resource
        df.loc[curr_ind, 'lifecycle_state'] = str(event.lifecycle_state)
        pkl.dump(df, open(path, 'wb'))

        # if a new task is activated or a case is completed
        if str(event.lifecycle_state) == 'EventType.TASK_ACTIVATE' or str(event.lifecycle_state) == 'EventType.COMPLETE_CASE':

            if not os.path.exists(path_freq_transition): # inital df_freq_transition
                all_cols = ['task', 'W_Complete application', 'W_Call after offers',
                            'W_Validate application', 'W_Call incomplete files',
                            'W_Handle leads', 'W_Assess potential fraud',
                            'W_Shortened completion', 'complete_case']
                num_cols = len(all_cols)
                initial_df_vals = np.zeros((num_cols - 2, num_cols))
                df_freq_transition = pd.DataFrame(initial_df_vals, columns=all_cols)
                for task_ind, task_ in enumerate(all_cols):
                    # print(task_, task_ind)
                    if (task_ind > 0) and (task_ind < num_cols - 1):
                        df_freq_transition.loc[task_ind - 1, 'task'] = task_
                pkl.dump(df_freq_transition, open(path_freq_transition, 'wb'))
            else:
                df_freq_transition = pkl.load(open(path_freq_transition, 'rb'))

            if str(event.lifecycle_state) == 'EventType.TASK_ACTIVATE':
                prev_lifecycle = df.loc[df.index[-2],'lifecycle_state']
                if prev_lifecycle == 'EventType.COMPLETE_TASK':
                    prev_task = df.loc[df.index[-2], 'task']
                    curr_task = df.loc[df.index[-1], 'task']
                    df_freq_transition.loc[df_freq_transition['task']==prev_task,curr_task] = df_freq_transition.loc[df_freq_transition['task']==prev_task,curr_task].item() + 1
                    pkl.dump(df_freq_transition, open(path_freq_transition, 'wb'))


            elif str(event.lifecycle_state) == 'EventType.COMPLETE_CASE':
                prev_task = df.loc[df.index[-2], 'task']
                df_freq_transition.loc[df_freq_transition['task']==prev_task,'complete_case'] = df_freq_transition.loc[df_freq_transition['task']==prev_task,'complete_case'].item() + 1
                pkl.dump(df_freq_transition, open(path_freq_transition, 'wb'))




        ## Create service mean and var df per combination of task and resource


        mean_path = './data/pd_mean_var_path_' + str(curr_num) + '.pkl'

        if not os.path.exists(mean_path):
            df_mean_var = pd.DataFrame(columns = ['resource'])

        else:
            df_mean_var = pkl.load(open(mean_path, 'rb'))

        if str(event.lifecycle_state) == 'EventType.COMPLETE_TASK': # if a task just completed
            resource = event.resource  # Give the current resource type
            task = str(event.task)  # Give the current task type
            # The index of the task start time event
            start_ind = df.loc[(df['task'] == task) & (df['lifecycle_state'] == 'EventType.START_TASK'), :].index[-1]
            # Computing the service time
            ser_time = event.timestamp- df.loc[start_ind,'timestamp']
            curr_resources = df_mean_var['resource'].unique() # All possible resources that were
            if resource in curr_resources:  # if the current resource already been added in the past
                get_ind = df_mean_var.loc[df_mean_var['resource']==resource,:].index[0] # if so, find its row
            else:
                get_ind = df_mean_var.shape[0]
                df_mean_var.loc[get_ind, 'resource'] = resource

            if 'count_' + task in  df_mean_var.columns:  # if this column exists
                get_count_value = df_mean_var.loc[get_ind, 'count_' + task]  # get the count so far
            else:
                get_count_value = 0
            if get_count_value > 0:  # if the count already took place
                get_mean_value = df_mean_var.loc[get_ind, 'mean_' + task]  # get the mean service time so far
                get_var_value = df_mean_var.loc[get_ind, 'var_' + task]  # get the service variance so far
                curr_mean = (get_count_value*get_mean_value+ser_time)/(get_count_value+1) # compute the new service mean
                if get_count_value > 1:  # For computing variance we need two values at least
                    squer_sum = get_var_value*(get_count_value-1)+get_count_value*get_mean_value**2 # updating variance
                    get_count_value += 1  # updating count
                    curr_var = (squer_sum+ser_time**2-get_count_value*curr_mean**2)/(get_count_value-1) # updating variance

                else:  # if this is the second time we get this combination (task, resource)
                    # the variace is updated accordingly
                    get_count_value = 2 # the count must be 2
                    curr_var = (get_mean_value-curr_mean)**2+(ser_time-curr_mean)**2 # by defintion
            else:
                get_count_value = 1
                curr_mean = ser_time
                curr_var = 0  # it is not really zero but not defined yet.


            # Updating df_mean_var table
            df_mean_var.loc[get_ind, 'mean_' + task] = curr_mean
            df_mean_var.loc[get_ind, 'var_' + task] = curr_var
            df_mean_var.loc[get_ind, 'count_' + task] = get_count_value
            pkl.dump(df_mean_var, open(mean_path, 'wb'))


def func1(x1,x2):
    return -((x1-3)**2+(x2-2)**2)+3

def get_curr_val(a1,a2,a3,a4,a5, curr_num, file_name):

    my_planner = MyPlanner()
    simulator = Simulator(my_planner, file_name)
    result = simulator.run(a1,a2,a3,a4,a5, curr_num)
    return result[0]




def main(args):


    for ind in range(args.num_iter):

        a1 = np.random.uniform(5,15)
        a2 = np.random.uniform(0, 10)
        a3 = np.random.uniform(0, 10)
        a4 = np.random.uniform(0, 10)
        a5 = np.random.uniform(5, 15)

        a1 = 10.879914
        a2 = 0.475911
        a3 = 1.456346
        a4 = 0.928605
        a5 = 8.479268

        a1 = 10.
        a2 = 1.
        a3 = 1.
        a4 = 1.
        a5 = 1.
        print(a1, a2, a3, a4, a5)

        curr_num = np.random.randint(1, 10000000)
        print('curr_num: ', curr_num)

        curr_result1 = get_curr_val(a1, a2, a3, a4, a5, curr_num, 'BPI Challenge 2017 - instance.pickle')
        curr_result2 = get_curr_val(a1, a2, a3, a4, a5, curr_num,'BPI Challenge 2017 - instance 2.pickle')


        if os.path.exists(args.file_path):
            df = pkl.load(open(args.file_path, 'rb'))
            print('Already we have {} data points' .format(df.shape[0]))
        else:
            df = pd.DataFrame([])
            print('first time with the df')

        curr_ind = df.shape[0]
        df.loc[curr_ind, 'a1'] = a1
        df.loc[curr_ind, 'a2'] = a2
        df.loc[curr_ind, 'a3'] = a3
        df.loc[curr_ind, 'a4'] = a4
        df.loc[curr_ind, 'a5'] = a5
        df.loc[curr_ind, 'result1'] = curr_result1
        df.loc[curr_ind, 'result2'] = curr_result2
        df.loc[curr_ind, 'result_tot'] = curr_result1 + curr_result2
        df.loc[curr_ind, 'curr_num'] = curr_num
        df.loc[curr_ind, 'data_example'] = 'both'

        pkl.dump(df, open(args.file_path, 'wb'))

        print(df)





def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, help='which settings are used', default='Result_table_4.pkl')
    parser.add_argument('--num_iter', type=int, help='how many iterations we run', default=5)
    args = parser.parse_args(argv)

    return args

if __name__ == "__main__":

    args = parse_arguments(sys.argv[1:])
    main(args)