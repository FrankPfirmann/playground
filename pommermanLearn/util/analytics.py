import time as T

class Stopwatch:
    def __init__(self, start=False):
        self.is_running=False
        self.start_time=self._get_timestamp()
        self.stop_time=self._get_timestamp()

        if start:
            self.start()

    def start(self):
        """
        Start the stopwatch if it is not already running.
        """
        assert not self.is_running

        self.start_time=self._get_timestamp()
        self.is_running=True

    def stop(self):
        """
        Stop the running stopwatch and return the time since start.

        :return: The time elapsed since starting the stopwatch in seconds
        """
        assert self.is_running

        self.stop_time=self._get_timestamp()
        self.is_running=False

        return self.elapsed()
    
    def elapsed(self):
        """
        Return the elapsed time measured by the stopwatch.

        :return: The time elapsed measured by the stopwatch in seconds.
        """
        if self.is_running:
            return self._get_timestamp() - self.start_time
        else:
            return self.stop_time - self.start_time

    def _get_timestamp(self):
        """
        Return a timestamp representing the current time.
        """
        return T.time()