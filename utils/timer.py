import time

class StepTimer:

    def __init__(self, total_steps=None):
        self.Lauched = False
        self.LastStep = None
        self.TotalSteps = total_steps
        self.CurrentStep = 0

    def begin(self):
        self.Lauched = True
        self.LastStep = time.time()

    def step(self, prt=True):
        if not self.Lauched:
            raise ValueError("Timer has not be lauched!")

        self.CurrentStep += 1
        now_time = time.time()

        cycle_time = now_time- self.LastStep

        if prt:
            print("Time consuming: %.2f" % cycle_time)
            if self.TotalSteps is not None:
                remaining_time = cycle_time * (self.TotalSteps - self.CurrentStep)
                remaining_hour = remaining_time // 3600
                remaining_min = (remaining_time % 3600) // 60
                remaining_sec = (remaining_time % 60)
                print('time remains:  %02d:%02d:%02d'%(remaining_hour, remaining_min, remaining_sec))
                print("*"*50)

        self.LastStep = now_time