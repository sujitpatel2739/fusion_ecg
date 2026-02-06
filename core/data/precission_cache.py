from collections import OrderedDict

class PrecisionCache:
    def __init__(self, capacity: int):
        """
        capacity: Maximum number of batches to store in RAM.
        """
        self.cache = OrderedDict()
        self.capacity = capacity

    def get_batch(self, batch_no: int):
        """Returns the batch if it exists, otherwise returns None."""
        if batch_no not in self.cache:
            return None
        
        # Move to end (mark as most recently used)
        self.cache.move_to_end(batch_no)
        return self.cache[batch_no]

    def add_batch(self, batch_no: int, batch_data):
        """Adds a batch. Evicts the oldest if capacity is reached."""
        if batch_no in self.cache:
            self.cache.move_to_end(batch_no)
        self.cache[batch_no] = batch_data
        
        if len(self.cache) > self.capacity:
            # last=False pops the first item (the Least Recently Used)
            self.cache.popitem(last=False)