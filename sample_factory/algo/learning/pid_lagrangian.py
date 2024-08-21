from collections import deque


class PIDLagrangian:
    def __init__(self, kp, ki, kd, target, error_history_length, ema_alpha, max_penalty=float('inf')):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target = target
        self.max_penalty = max_penalty
        self.integral = 0
        self.previous_error = 0
        self.error_history = deque([0.0] * error_history_length, maxlen=error_history_length)
        self.ema_alpha = ema_alpha
        self.delta_d = 0.0

    def update(self, current_value):
        # Calculate current error
        error = current_value - self.target

        # Update integral of error
        self.integral += error

        # Calculate derivative of error
        self.error_history.append(error)
        derivative = (error - self.error_history[0]) / len(self.error_history)

        # Exponentially Weighted Moving Average for the derivative term
        self.delta_d = self.delta_d * self.ema_alpha + (1 - self.ema_alpha) * derivative

        # Calculate PID output
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * self.delta_d)

        # Clamp output to max_penalty
        if self.max_penalty is not None:
            output = max(min(output, self.max_penalty), 0)

        # Update previous error
        self.previous_error = error

        return output
