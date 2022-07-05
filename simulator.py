from enum import Enum, auto
from datetime import datetime, timedelta
import random
import pickle5 as pickle
from abc import ABC, abstractmethod


RUNNING_TIME = 24*365


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

    def run(self, a1,a2,a3,a4,a5):
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
                self.planner.report(Event(event.task.case_id, None, self.now, None, EventType.CASE_ARRIVAL))
                # add new task
                self.planner.report(Event(event.task.case_id, event.task, self.now, None, EventType.TASK_ACTIVATE))
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
                self.planner.report(Event(event.task.case_id, event.task, self.now, event.resource, EventType.START_TASK))
                # create a complete event for task
                t = self.now + self.problem.processing_time(event.task, event.resource)
                self.events.append((t, SimulationEvent(EventType.COMPLETE_TASK, t, event.task, event.resource)))
                self.events.sort()
                # set resource to busy
                del self.reserved_resources[event.resource]
                self.busy_resources[event.resource] = (event.task, self.now)

            # if e is a complete event:
            elif event.event_type == EventType.COMPLETE_TASK:
                self.planner.report(Event(event.task.case_id, event.task, self.now, event.resource, EventType.COMPLETE_TASK))
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
                    self.planner.report(Event(event.task.case_id, next_task, self.now, None, EventType.TASK_ACTIVATE))
                    self.unassigned_tasks[next_task.id] = next_task
                    self.busy_cases[event.task.case_id].append(next_task.id)
                if len(self.busy_cases[event.task.case_id]) == 0:
                    self.planner.report(Event(event.task.case_id, None, self.now, None, EventType.COMPLETE_CASE))
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
                    assignments = self.planner.plan(self.available_resources.copy(), list(self.unassigned_tasks.values()), self.problem_resource_pool, a1,a2,a3,a4,a5)
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
