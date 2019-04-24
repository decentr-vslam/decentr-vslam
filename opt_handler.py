class OptHandle:
    
    def __init__(self, robot_num):
        self.optimized_groups = list()
        self.futures = list()
        self.group_assignment = list()
        self.between_robot_links = list()

        # assign group structure
        for i in range(robot_num):
            opt_group = dict()
            opt_group['members'] = [i]
            self.optimized_groups.append(opt_group)

            # future 
            future = dict()
            future['State'] = "running"
            future['Error'] = []
            future['OutputArguments'] = []
            self.futures.append(future)
            
        self.running = False