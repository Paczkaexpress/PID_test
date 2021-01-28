import control
import numpy as np
import matplotlib.pyplot as plt
import random

class PID:
    """
    Implements a control method: PID controller

    Controller features:
    Standard proportional part
    Integrator part implementes anti-windup protection
    Derivative part based on the position change instead of error. That gave a better performance when the position demand is changed
    Controller has demand limit implemented

    Controller is a single PID block desing. The controller can only measure the position change (position sensor)
    """
    # it is only possible to measure the position so based on that we need to design a conctroller  
    pid_err_sum = [0]
    demand = 0
    windup = 0
    pid_gain = [0]
    pid_gain_limited = [0]

    def __init__(self, kp, kd, ki, limit):
        self.pid_kp = kp
        self.pid_ki = ki
        self.pid_kd = kd
        self.limit = limit

    def control(self, pos_sensor: float, prev_pos_sensor: float):
        pos_error = self.demand - pos_sensor
        
        # calculate error sum 
        self.pid_err_sum.append(self.pid_err_sum[-1] + pos_error + self.windup)
        
        # calculate pid gain
        self.pid_gain.append(self.pid_kp * pos_error + self.pid_ki * (self.pid_err_sum[-1]) + self.pid_kd * (-(pos_sensor-prev_pos_sensor)))
        
        # real system response limits
        if(self.pid_gain[-1] > self.limit):
            self.pid_gain_limited.append(self.limit)
        elif(self.pid_gain[-1] < -self.limit):
            self.pid_gain_limited.append(-self.limit)
        else:
            self.pid_gain_limited.append(self.pid_gain[-1])

        # anti-windup protection
        self.windup = self.pid_gain_limited[-1] - self.pid_gain[-1]

    def set_demand(self, new_demand):
        """
        New position demand setter
        """
        self.demand = new_demand


class Sensor:
    """
    The sensor model simulated the real world behaviour

    It simulates the following effects:
        * quantisation
        * limited resolution / accuracy
        * measurement delay
        * system noise
        * limitted update rate
    """
    accuracy = 0
    noise_lvl = 0
    refresh_rate = 0
    measurements = [0, 0]
    time = []

    def __init__(self, accuracy, noise_lvl, delay, refresh_rate):
        self.accuracy = accuracy
        self.noise_lvl = noise_lvl
        self.refresh_rate = refresh_rate
        self.delay = delay

    def measure(self, sample: int, real_pos: list()) -> float:
        # limiting sensor measurement frequency
        if(sample % self.refresh_rate == 0):

            # Adding some measurement delay
            measurement = real_pos[-2] 

            # adding some random noise
            measurement = measurement + self.noise_lvl*(random.random()-0.5) 

            # limiting the system resolution
            measurement = float(int(measurement/self.accuracy))*self.accuracy 
            self.measurements.append(measurement)
            self.time.append(sample)

    def get_filtered(self, index, size):
        return sum(self.measurements[(-size+index):index])/size


class SpringDampMassModel:
    """
    Defines a mathematical model of the mass, spring damper system

    Mathematical model of the system is:
    A = [0 1;-k/m -b/m]
    B = [0; 1/m]
    C = [1 0]
    D = [0]

    Integration is performed using a square integration method. If the given method 
    is not satisfactory the trapezoidal, hyperbolic, or even 
    Runge-Kutta method can be implemented.  
    """
    pos = [0]
    vel = [0]
    acc = [0]
    dist = 0

    def __init__(self, mass: float, spring: float, damper: float, init_pos, init_vel, init_acc, pid):
        self.dt = dt
        self.mass = mass
        self.spring = spring
        self.damper = damper
        self.pos[0] = init_pos
        self.vel[0] = init_vel
        self.acc[0] = init_acc
        self.controller = pid 

    def update(self, dt):
        """
        Update function calcualtes the new state on the system based on the distorion value, controller demand,
        and previous system states.
        """
        acc_tmp = self.dist + self.controller.pid_gain_limited[-1] + \
            ((-self.damper/self.mass) * self.vel[-1]) + \
                ((-self.spring/self.mass) * self.pos[-1])
        vel_tmp = self.vel[-1] + acc_tmp * dt
        pos_tmp = self.pos[-1] + vel_tmp * dt

        self.acc.append(acc_tmp)
        self.vel.append(vel_tmp)
        self.pos.append(pos_tmp)

    def set_distorion(self, distortion):
        """
        Set distortion function injects distortion to the system
        """
        self.dist = distortion


if __name__ == "__main__":
    dt = 0.01
    end_time = 1500
    end_range = int(end_time / dt)
    time = range(0, end_range)
    dist = 0

    pid = PID(1, 6, 0.001, 1.5)
    sensor = Sensor(0.01, 0.05, 0.02, 10)
    model = SpringDampMassModel(1000, 1, 10, 0, 0, 0, pid)

    for t, i in enumerate(time):
        if(i == 0):
            continue
        model.update(dt)

        # system disturbance test
        if(t == 100/dt):
            model.set_distorion(1)
        if(t == 300/dt):
            model.set_distorion(0)
        
        # set a new position demand
        if(t == 500/dt):
            pid.set_demand(1)

        # set a new position demand with mechanical lock /  anti windup test
        if(t == 700/dt):
            pid.set_demand(-2)
        if(t > 700/dt and t < 1000/dt and model.pos[-1] < 0.8):
            model.vel[-1] = 0
            model.pos[-1] = model.pos[-2]

        sensor.measure(i, model.pos)
        pid.control(sensor.get_filtered(-1, 2), sensor.get_filtered(-2, 2))

    time = [t * dt for t in time]

    fix, ax = plt.subplots(2) 
    ax[0].plot(time[:len(model.pos)], model.pos)
    ax[0].set_title("Step Response")
    
    time_sensor = [t * dt for t in sensor.time]
    ax[0].plot(time_sensor, sensor.measurements[:len(time_sensor)])
    

    ax[1].plot(time[:len(pid.pid_gain)], pid.pid_gain)
    ax[1].plot(time[:len(pid.pid_gain_limited)], pid.pid_gain_limited)
    ax[1].set_title("PID gain")

    plt.show()