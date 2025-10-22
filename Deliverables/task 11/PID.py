class PID:
    def __init__(self, kp, ki, kd, setpoint=0.0, output_limits=(None, None)):
        """
        Initialize the PID controller with given coefficients and setpoint.

        :param kp: Proportional gain
        :param ki: Integral gain
        :param kd: Derivative gain
        :param setpoint: Desired setpoint
        :param output_limits: Tuple (min, max) limits for the output
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint

        self._integral = 0.0
        self._previous_error = 0.0

        self.output_limits = output_limits

    def reset(self):
        """
        Reset the PID controller history.
        """
        self._integral = 0.0
        self._previous_error = 0.0

    def compute(self, measurement, dt):
        """
        Compute the PID output using the PID formula.

        :param measurement: The current measurement of the process variable
        :param dt: Time interval since the last update
        :return: Control output
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive and non-zero")

        # Calculate error
        error = self.setpoint - measurement

        # Proportional term
        p = self.kp * error

        # Integral term
        self._integral += error * dt
        i = self.ki * self._integral

        # Derivative term
        derivative = (error - self._previous_error) / dt
        d = self.kd * derivative

        # Compute the output
        output = p + i + d

        # Apply output limits
        min_output, max_output = self.output_limits
        if min_output is not None:
            output = max(min_output, output)
        if max_output is not None:
            output = min(max_output, output)

        # Save error for next derivative calculation
        self._previous_error = error

        return output
