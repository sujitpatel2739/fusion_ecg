from collections import OrderedDict

class PrecisionCache:
    def __init__(self, capacity: int):
        """
        capacity: Maximum number of batches to store in RAM.
        """
        self.cache = OrderedDict()
        self.capacity = capacity

    def get_batch(self, key):
        """Returns the batch if it exists, otherwise returns None."""
        if key not in self.cache:
            return None
        
        # Move to end (mark as most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]

    def add_batch(self, key, batch_data):
        """Adds a batch. Evicts the oldest if capacity is reached."""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = batch_data
        
        if len(self.cache) > self.capacity:
            # last=False pops the first item (the Least Recently Used)
            self.cache.popitem(last=False)