from enum import Enum, auto
from datetime import datetime, timedelta
import random
import pickle5 as pickle
from abc import ABC, abstractmethod


RUNNING_TIME = 24*3


class Event:

    initial_time = datetime(2020, 1, 1)
    time_format = "%Y-%m-%d %H:%M:%S.%f"

    def __init__(self, case_id, task, timestamp, resource, lifecycle_state):
        self.case_id = case_id
        self.task = task
        self.timestamp = timestamp
        self.resource = resource
        self.lifecycle_state = lifecycle_state

    def __str__(self):
        t = (self.initial_time + timedelta(hours=self.timestamp)).strftime(self.time_format)
        return str(self.case_id) + "\t" + str(self.task) + "\t" + t + "\t" + str(self.resource) + "\t" + str(self.lifecycle_state)


class Task:

    def __init__(self, task_id, case_id, task_type):
        self.id = task_id
        self.case_id = case_id
        self.task_type = task_type

    def __lt__(self, other):
        return self.id < other.id

    def __str__(self):
        return self.task_type


class Problem(ABC):

    @property
    @abstractmethod
    def resources(self):
        raise NotImplementedError

    @property
    def resource_weights(self):
        return self._resource_weights

    @resource_weights.setter
    def resource_weights(self, value):
        self._resource_weights = value

    @property
    def schedule(self):
        return self._schedule

    @schedule.setter
    def schedule(self, value):
        self._schedule = value

    @property
    @abstractmethod
    def task_types(self):
        raise NotImplementedError

    @abstractmethod
    def sample_initial_task_type(self):
        raise NotImplementedError

    def resource_pool(self, task_type):
        return self.resources

    def __init__(self):
        self.next_case_id = 0
        self.cases = dict()  # case_id -> (arrival_time, initial_task)
        self._resource_weights = [1]*len(self.resources)
        self._schedule = [len(self.resources)]
        self._task_processing_times = dict()
        self._task_next_tasks = dict()

    def from_generator(self, duration):
        now = 0
        next_case_id = 0
        next_task_id = 0
        unfinished_tasks = []
        # Instantiate cases at the interarrival time for the duration.
        # Generate the first task for each case, without processing times and next tasks, add them to the unfinished tasks.
        while now < duration:
            at = now + self.interarrival_time_sample()
            initial_task_type = self.sample_initial_task_type()
            task = Task(next_task_id, next_case_id, initial_task_type)
            next_task_id += 1
            unfinished_tasks.append(task)
            self.cases[next_case_id] = (at, task)
            next_case_id += 1
            now = at
        # Finish the tasks by:
        # 1. generating the processing times.
        # 2. generating the next tasks, without processing times and next tasks, add them to the unfinished tasks.
        while len(unfinished_tasks) > 0:
            task = unfinished_tasks.pop(0)
            for r in self.resource_pool(task.task_type):
                pt = self.processing_time_sample(r, task)
                if task not in self._task_processing_times:
                    self._task_processing_times[task] = dict()
                self._task_processing_times[task][r] = pt
            for tt in self.next_task_types_sample(task):
                new_task = Task(next_task_id, task.case_id, tt)
                next_task_id += 1
                unfinished_tasks.append(new_task)
                if task not in self._task_next_tasks:
                    self._task_next_tasks[task] = []
                self._task_next_tasks[task].append(new_task)
        return self

    @classmethod
    def from_file(cls, filename):
        with open(filename, 'rb') as handle:
            instance = pickle.load(handle)
        return instance

    def save_instance(self, filename):
        with open(filename, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @abstractmethod
    def processing_time_sample(self, resource, task):
        raise NotImplementedError

    @abstractmethod
    def interarrival_time_sample(self):
        raise NotImplementedError

    def next_task_types_sample(self, task):
        return []

    def restart(self):
        self.next_case_id = 0

    def next_case(self):
        try:
            (arrival_time, initial_task) = self.cases[self.next_case_id]
            self.next_case_id += 1
            return arrival_time, initial_task
        except KeyError:
            return None

    def next_tasks(self, task):
        if task in self._task_next_tasks:
            return self._task_next_tasks[task]
        else:
            return []

    def processing_time(self, task, resource):
        return self._task_processing_times[task][resource]


class MinedProblem(Problem):

    resources = []
    task_types = []

    def __init__(self):
        super().__init__()
        self.initial_task_distribution = []
        self.next_task_distribution = dict()
        self.mean_interarrival_time = 0
        self.resource_pools = dict()
        self.processing_time_distribution = dict()

    def sample_initial_task_type(self):
        rd = random.random()
        rs = 0
        for (p, tt) in self.initial_task_distribution:
            rs += p
            if rd < rs:
                return tt
        print("WARNING: the probabilities of initial tasks do not add up to 1.0")
        return self.initial_task_distribution[0]

    def resource_pool(self, task_type):
        return self.resource_pools[task_type]

    def interarrival_time_sample(self):
        return random.expovariate(1/self.mean_interarrival_time)

    def next_task_types_sample(self, task):
        rd = random.random()
        rs = 0
        for (p, tt) in self.next_task_distribution[task.task_type]:
            rs += p
            if rd < rs:
                if tt is None:
                    return []
                else:
                    return [tt]
        print("WARNING: the probabilities of next tasks do not add up to 1.0")
        if self.next_task_distribution[0][1] is None:
            return []
        else:
            return [self.next_task_distribution[0][1]]

    def processing_time_sample(self, resource, task):
        (mu, sigma) = self.processing_time_distribution[(task.task_type, resource)]
        pt = random.gauss(mu, sigma)
        while pt < 0:  # We do not allow negative values for processing time.
            pt = random.gauss(mu, sigma)
        return pt

    @classmethod
    def generator_from_file(cls, filename):
        o = MinedProblem()
        with open(filename, 'rb') as handle:
            o.resources = pickle.load(handle)
            o.task_types = pickle.load(handle)
            o.initial_task_distribution = pickle.load(handle)
            o.next_task_distribution = pickle.load(handle)
            o.mean_interarrival_time = pickle.load(handle)
            o.resource_pools = pickle.load(handle)
            o.processing_time_distribution = pickle.load(handle)
            o.resource_weights = pickle.load(handle)
            o.schedule = pickle.load(handle)
        return o

    def save_generator(self, filename):
        with open(filename, 'wb') as handle:
            pickle.dump(self.resources, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.task_types, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.initial_task_distribution, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.next_task_distribution, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.mean_interarrival_time, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.resource_pools, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.processing_time_distribution, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.resource_weights, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.schedule, handle, protocol=pickle.HIGHEST_PROTOCOL)


class EventType(Enum):
    CASE_ARRIVAL = auto()
    START_TASK = auto()
    COMPLETE_TASK = auto()
    PLAN_TASKS = auto()
    TASK_ACTIVATE = auto()
    TASK_PLANNED = auto()
    COMPLETE_CASE = auto()
    SCHEDULE_RESOURCES = auto()


class TimeUnit(Enum):
    SECONDS = auto()
    MINUTES = auto()
    HOURS = auto()
    DAYS = auto()


class SimulationEvent:
    def __init__(self, event_type, moment, task, resource=None, nr_tasks=0, nr_resources=0):
        self.event_type = event_type
        self.moment = moment
        self.task = task
        self.resource = resource
        self.nr_tasks = nr_tasks
        self.nr_resources = nr_resources

    def __lt__(self, other):
        return self.moment < other.moment

    def __str__(self):
        return str(self.event_type) + "\t(" + str(round(self.moment, 2)) + ")\t" + str(self.task) + "," + str(self.resource)


class Simulator:
    def __init__(self, planner, instance_file="BPI Challenge 2017 - instance.pickle"):
        print(instance_file)
        self.events = []

        self.unassigned_tasks = dict()
        self.assigned_tasks = dict()
        self.available_resources = set()
        self.away_resources = []
        self.away_resources_weights = []
        self.busy_resources = dict()
        self.busy_cases = dict()
        self.reserved_resources = dict()
        self.now = 0

        self.finalized_cases = 0
        self.total_cycle_time = 0
        self.case_start_times = dict()

        self.planner = planner
        self.problem = MinedProblem.from_file(instance_file)
        self.problem_resource_pool = self.problem.resource_pools

        self.init_simulation()

    def init_simulation(self):
        # set all resources to available
        for r in self.problem.resources:
            self.available_resources.add(r)

        # generate resource scheduling event to start the schedule
        self.events.append((0, SimulationEvent(EventType.SCHEDULE_RESOURCES, 0, None)))

        # reset the problem
        self.problem.restart()

        # generate arrival event for the first task of the first case
        (t, task) = self.problem.next_case()
        self.events.append((t, SimulationEvent(EventType.CASE_ARRIVAL, t, task)))

    def desired_nr_resources(self):
        return self.problem.schedule[int(self.now % len(self.problem.schedule))]

    def working_nr_resources(self):
        return len(self.available_resources) + len(self.busy_resources) + len(self.reserved_resources)

    def run(self, a1,a2,a3,a4,a5, curr_num, df, df_freq_transition, df_mean_var):
        # repeat until the end of the simulation time:
        while self.now <= RUNNING_TIME:
            # get the first event e from the events
            event = self.events.pop(0)
            # t = time of e
            self.now = event[0]
            event = event[1]

            # if e is an arrival event:
            if event.event_type == EventType.CASE_ARRIVAL:
                self.case_start_times[event.task.case_id] = self.now
                df, df_freq_transition, df_mean_var =  self.planner.report(Event(event.task.case_id, None, self.now, None, EventType.CASE_ARRIVAL), curr_num, df, df_freq_transition, df_mean_var)
                # add new task
                df, df_freq_transition, df_mean_var = self.planner.report(Event(event.task.case_id, event.task, self.now, None, EventType.TASK_ACTIVATE), curr_num, df, df_freq_transition, df_mean_var)
                self.unassigned_tasks[event.task.id] = event.task
                self.busy_cases[event.task.case_id] = [event.task.id]
                # generate a new planning event to start planning now for the new task
                self.events.append((self.now, SimulationEvent(EventType.PLAN_TASKS, self.now, None, nr_tasks=len(self.unassigned_tasks), nr_resources=len(self.available_resources))))
                # generate a new arrival event for the first task of the next case
                (t, task) = self.problem.next_case()
                self.events.append((t, SimulationEvent(EventType.CASE_ARRIVAL, t, task)))
                self.events.sort()

            # if e is a start event:
            elif event.event_type == EventType.START_TASK:
                df, df_freq_transition, df_mean_var = self.planner.report(Event(event.task.case_id, event.task, self.now, event.resource, EventType.START_TASK), curr_num, df, df_freq_transition, df_mean_var)
                # create a complete event for task
                t = self.now + self.problem.processing_time(event.task, event.resource)
                self.events.append((t, SimulationEvent(EventType.COMPLETE_TASK, t, event.task, event.resource)))
                self.events.sort()
                # set resource to busy
                del self.reserved_resources[event.resource]
                self.busy_resources[event.resource] = (event.task, self.now)

            # if e is a complete event:
            elif event.event_type == EventType.COMPLETE_TASK:
                df, df_freq_transition, df_mean_var =  self.planner.report(Event(event.task.case_id, event.task, self.now, event.resource, EventType.COMPLETE_TASK), curr_num, df, df_freq_transition, df_mean_var)
                # set resource to available, if it is still desired, otherwise set it to away
                del self.busy_resources[event.resource]
                if self.working_nr_resources() <= self.desired_nr_resources():
                    self.available_resources.add(event.resource)
                else:
                    self.away_resources.append(event.resource)
                    self.away_resources_weights.append(self.problem.resource_weights[self.problem.resources.index(event.resource)])
                # remove task from assigned tasks
                del self.assigned_tasks[event.task.id]
                self.busy_cases[event.task.case_id].remove(event.task.id)
                # generate unassigned tasks for each next task
                for next_task in self.problem.next_tasks(event.task):
                    df, df_freq_transition, df_mean_var = self.planner.report(Event(event.task.case_id, next_task, self.now, None, EventType.TASK_ACTIVATE), curr_num, df, df_freq_transition, df_mean_var)
                    self.unassigned_tasks[next_task.id] = next_task
                    self.busy_cases[event.task.case_id].append(next_task.id)
                if len(self.busy_cases[event.task.case_id]) == 0:
                    df, df_freq_transition, df_mean_var =  self.planner.report(Event(event.task.case_id, None, self.now, None, EventType.COMPLETE_CASE), curr_num, df, df_freq_transition, df_mean_var)
                    self.events.append((self.now, SimulationEvent(EventType.COMPLETE_CASE, self.now, event.task)))
                # generate a new planning event to start planning now for the newly available resource and next tasks
                self.events.append((self.now, SimulationEvent(EventType.PLAN_TASKS, self.now, None, nr_tasks=len(self.unassigned_tasks), nr_resources=len(self.available_resources))))
                self.events.sort()

            # if e is a schedule resources event: move resources between available/away,
            # depending to how many resources should be available according to the schedule.
            elif event.event_type == EventType.SCHEDULE_RESOURCES:
                assert self.working_nr_resources() + len(self.away_resources) == len(self.problem.resources)  # the number of resources must be constant
                assert len(self.problem.resources) == len(self.problem.resource_weights)  # each resource must have a resource weight
                assert len(self.away_resources) == len(self.away_resources_weights)  # each away resource must have a resource weight
                if len(self.away_resources) > 0:  # for each away resource, the resource weight must be taken from the problem resource weights
                    i = random.randrange(len(self.away_resources))
                    assert self.away_resources_weights[i] == self.problem.resource_weights[self.problem.resources.index(self.away_resources[i])]
                required_resources = self.desired_nr_resources() - self.working_nr_resources()
                if required_resources > 0:
                    # if there are not enough resources working
                    # randomly select away resources to work, as many as required
                    for i in range(required_resources):
                        random_resource = random.choices(self.away_resources, self.away_resources_weights)[0]
                        # remove them from away and add them to available resources
                        away_resource_i = self.away_resources.index(random_resource)
                        del self.away_resources[away_resource_i]
                        del self.away_resources_weights[away_resource_i]
                        self.available_resources.add(random_resource)
                    # generate a new planning event to put them to work
                    self.events.append((self.now, SimulationEvent(EventType.PLAN_TASKS, self.now, None, nr_tasks=len(self.unassigned_tasks), nr_resources=len(self.available_resources))))
                    self.events.sort()
                elif required_resources < 0:
                    # if there are too many resources working
                    # remove as many as possible, i.e. min(available_resources, -required_resources)
                    nr_resources_to_remove = min(len(self.available_resources), -required_resources)
                    resources_to_remove = random.sample(self.available_resources, nr_resources_to_remove)
                    for r in resources_to_remove:
                        # remove them from the available resources
                        self.available_resources.remove(r)
                        # add them to the away resources
                        self.away_resources.append(r)
                        self.away_resources_weights.append(self.problem.resource_weights[self.problem.resources.index(r)])
                # plan the next resource schedule event
                self.events.append((self.now + 1, SimulationEvent(EventType.SCHEDULE_RESOURCES, self.now + 1, None)))

            # if e is a planning event: do assignment
            elif event.event_type == EventType.PLAN_TASKS:
                # there only is an assignment if there are free resources and tasks
                if len(self.unassigned_tasks) > 0 and len(self.available_resources) > 0:
                    assignments = self.planner.plan(self.available_resources.copy(), list(self.unassigned_tasks.values()), self.problem_resource_pool, a1,a2,a3,a4,a5, curr_num, df, df_freq_transition, df_mean_var)
                    # for each newly assigned task:
                    moment = self.now
                    for (task, resource) in assignments:
                        if task not in self.unassigned_tasks.values():
                            return None, "ERROR: trying to assign a task that is not in the unassigned_tasks."
                        if resource not in self.available_resources:
                            return None, "ERROR: trying to assign a resource that is not in available_resources."
                        if resource not in self.problem_resource_pool[task.task_type]:
                            return None, "ERROR: trying to assign a resource to a task that is not in its resource pool."
                        # create start event for task
                        self.events.append((moment, SimulationEvent(EventType.START_TASK, moment, task, resource)))
                        # assign task
                        del self.unassigned_tasks[task.id]
                        self.assigned_tasks[task.id] = (task, resource, moment)
                        # reserve resource
                        self.available_resources.remove(resource)
                        self.reserved_resources[resource] = (event.task, moment)
                    self.events.sort()

            # if e is a complete case event: add to the number of completed cases
            elif event.event_type == EventType.COMPLETE_CASE:
                self.total_cycle_time += self.now - self.case_start_times[event.task.case_id]
                self.finalized_cases += 1

        unfinished_cases = 0
        for busy_tasks in self.busy_cases.values():
            if len(busy_tasks) > 0:
                if busy_tasks[0] in self.unassigned_tasks:
                    busy_case_id = self.unassigned_tasks[busy_tasks[0]].case_id
                else:
                    busy_case_id = self.assigned_tasks[busy_tasks[0]][0].case_id
                if busy_case_id in self.case_start_times:
                    start_time = self.case_start_times[busy_case_id]
                    if start_time <= RUNNING_TIME:
                        self.total_cycle_time += RUNNING_TIME - start_time
                        self.finalized_cases += 1
                        unfinished_cases += 1

        return self.total_cycle_time / self.finalized_cases, "COMPLETED: you completed a full year of simulated customer cases."


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
        if df_trans.loc[df_trans['task'] == task_, :].shape[0]>0:
            curr_ind = df_trans.loc[df_trans['task'] == task_, :].index[0]
            if df_trans.iloc[curr_ind, 1:].sum()>0:
                return df_trans.iloc[curr_ind, -1] / df_trans.iloc[curr_ind, 1:].sum()
            else:
                return 0
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

        if (df_mean_var.shape[0] > 0) & (resource in list(df_mean_var['resource'])):  # if df_mean_var initiated and if we have knowegle about resource
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

    def plan(self, available_resources, unassigned_tasks, resource_pool, a1,a2,a3,a4,a5, curr_num, df,df_freq_transition, df_mean_var):

        # path_freq_transition = './data/freq_transition_path_' + str(curr_num) + '.pkl'
        # mean_path = './data/pd_mean_var_path_' + str(curr_num) + '.pkl'
        # path = './data/pd_path_' + str(curr_num) + '.pkl'
        #
        # if os.path.exists(path):
        #
        #     df = pkl.load(open(path, 'rb'))
        #
        #     curr_df_status = df.index[-1]
        # else:
        #     curr_df_status = 0
        # # print(df.index[-1])
        #
        #
        # if not os.path.exists(mean_path):
        #     df_mean_var = pd.DataFrame(columns=['resource'])
        # else:
        #     df_mean_var = pkl.load(open(mean_path, 'rb'))

        curr_df_status = df.index[-1]

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

                    # if os.path.exists(path_freq_transition):
                    if df_freq_transition.shape[0]> 0:
                        # df_freq_transition = pkl.load(open(path_freq_transition, 'rb'))
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

                    # if os.path.exists(path_freq_transition):
                    if df_freq_transition.shape[0]>0:
                        # df_freq_transition = pkl.load(open(path_freq_transition, 'rb'))
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


    def report(self, event, curr_num, df, df_freq_transition, df_mean_var):

        # path = './data/pd_path_' +str(curr_num)+'.pkl'
        #
        # path_freq_transition =  './data/freq_transition_path_' +str(curr_num)+'.pkl'


        # if not os.path.exists(path):
        #     df = pd.DataFrame([])
        # else:
        #     df = pkl.load(open(path, 'rb'))
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
        # pkl.dump(df, open(path, 'wb'))

        # if a new task is activated or a case is completed
        if str(event.lifecycle_state) == 'EventType.TASK_ACTIVATE' or str(event.lifecycle_state) == 'EventType.COMPLETE_CASE':

            # if not os.path.exists(path_freq_transition): # inital df_freq_transition
            #     all_cols = ['task', 'W_Complete application', 'W_Call after offers',
            #                 'W_Validate application', 'W_Call incomplete files',
            #                 'W_Handle leads', 'W_Assess potential fraud',
            #                 'W_Shortened completion', 'complete_case']
            #     num_cols = len(all_cols)
            #     initial_df_vals = np.zeros((num_cols - 2, num_cols))
            #     df_freq_transition = pd.DataFrame(initial_df_vals, columns=all_cols)
            #     for task_ind, task_ in enumerate(all_cols):
            #         # print(task_, task_ind)
            #         if (task_ind > 0) and (task_ind < num_cols - 1):
            #             df_freq_transition.loc[task_ind - 1, 'task'] = task_
            #     pkl.dump(df_freq_transition, open(path_freq_transition, 'wb'))
            # else:
            #     df_freq_transition = pkl.load(open(path_freq_transition, 'rb'))

            if str(event.lifecycle_state) == 'EventType.TASK_ACTIVATE':
                prev_lifecycle = df.loc[df.index[-2],'lifecycle_state']
                if prev_lifecycle == 'EventType.COMPLETE_TASK':
                    prev_task = df.loc[df.index[-2], 'task']
                    curr_task = df.loc[df.index[-1], 'task']
                    df_freq_transition.loc[df_freq_transition['task']==prev_task,curr_task] = df_freq_transition.loc[df_freq_transition['task']==prev_task,curr_task].item() + 1
                    # pkl.dump(df_freq_transition, open(path_freq_transition, 'wb'))


            elif str(event.lifecycle_state) == 'EventType.COMPLETE_CASE':
                prev_task = df.loc[df.index[-2], 'task']
                df_freq_transition.loc[df_freq_transition['task']==prev_task,'complete_case'] = df_freq_transition.loc[df_freq_transition['task']==prev_task,'complete_case'].item() + 1
                # pkl.dump(df_freq_transition, open(path_freq_transition, 'wb'))




        ## Create service mean and var df per combination of task and resource


        mean_path = './data/pd_mean_var_path_' + str(curr_num) + '.pkl'

        # if not os.path.exists(mean_path):
        #     df_mean_var = pd.DataFrame(columns = ['resource'])
        #
        # else:
        #     df_mean_var = pkl.load(open(mean_path, 'rb'))

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
            # pkl.dump(df_mean_var, open(mean_path, 'wb'))

        return (df, df_freq_transition, df_mean_var)


def func1(x1,x2):
    return -((x1-3)**2+(x2-2)**2)+3

def get_curr_val(a1,a2,a3,a4,a5, curr_num, df, df_freq_transition, df_mean_var, file_name):



    my_planner = MyPlanner()
    simulator = Simulator(my_planner, file_name)
    result = simulator.run(a1,a2,a3,a4,a5, curr_num, df, df_freq_transition, df_mean_var)
    return result[0]




def main(args):

    df1 = pd.DataFrame([])
    df_mean_var = pd.DataFrame([], columns=['resource'])

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




    for ind in range(args.num_iter):

        a1 = np.random.uniform(5,15)
        a2 = np.random.uniform(0, 2)
        a3 = np.random.uniform(0, 2)
        a4 = np.random.uniform(0, 2)
        a5 = np.random.uniform(5, 15)
        print(a1, a2, a3, a4, a5)

        curr_num = np.random.randint(1, 10000000)
        print('curr_num: ', curr_num)

        curr_result1 = get_curr_val(a1, a2, a3, a4, a5, curr_num, df1, df_freq_transition, df_mean_var, 'BPI Challenge 2017 - instance.pickle')
        # curr_result2 = get_curr_val(a1, a2, a3, a4, a5, curr_num,'BPI Challenge 2017 - instance 2.pickle')


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
        # df.loc[curr_ind, 'result2'] = curr_result2
        # df.loc[curr_ind, 'result_tot'] = curr_result1 + curr_result2
        df.loc[curr_ind, 'curr_num'] = curr_num
        df.loc[curr_ind, 'data_example'] = 'both'

        pkl.dump(df, open(args.file_path, 'wb'))

        print(df)





def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, help='which settings are used', default='Result_table_3.pkl')
    parser.add_argument('--num_iter', type=int, help='how many iterations we run', default=5)
    args = parser.parse_args(argv)

    return args

if __name__ == "__main__":

    args = parse_arguments(sys.argv[1:])
    main(args)