"""
Classes for objects in the warehouse, i.e. agents, bins, and staging areas.
"""
import copy
import random


NO_ITEM = 0


class WarehouseObject:
    """
    Base class for warehouse object: has position and status.
    Agents, bins and staging areas are warehouse objects.

    Parameters
    ----------
    data : dict
        A dictionary containining the information about the object, as it is
        stored in the json file. Must have 'position' and 'status' keys.
    """
    def __init__(self, data):
        self.position = tuple(data['position'])
        self.status = data['status']
        self.initial_position = copy.copy(self.position)
        self.initial_status = copy.copy(self.status)

    def reset(self):
        """
        Returns this object to its initial state, i.e. its initial position
        and status.
        """
        self.position = copy.copy(self.initial_position)
        self.status = copy.copy(self.initial_status)

    def is_empty(self):
        """
        Checks whether
        """
        return all(item == NO_ITEM for item in self.status)

    def free(self, slot):
        """
        Checks whether the specified slot is free.

        Parameters
        ----------
        slot : int
            The slot to check.

        Returns
        -------
        bool
            True if there is no item in the slot.
        """
        return self.status[slot] == NO_ITEM

    def pick(self, slot):
        """
        Picks the item from the specified slot. The slot must be non-empty.
        After picking, the slot is empty.

        Parameters
        ----------
        slot : int
            The slot to pick from.

        Returns
        -------
        item : int
            The item that was in the slot.
        """
        assert self.status[slot] != NO_ITEM
        item = self.status[slot]
        self.status[slot] = NO_ITEM
        return item

    def put(self, slot, item):
        """
        Puts the item into the specified slot. The slot must be empty.
        After putting, the item is in the slot.

        Parameters
        ----------
        slot : int
            The slot to put into.
        item : int
            The item to put into the slot.
        """
        assert self.status[slot] == NO_ITEM
        self.status[slot] = item


class Agent(WarehouseObject):
    """
    The agent.
    """
    pass


class Bin(WarehouseObject):
    """
    A WarehouseObject that additionally has a list of access spots.
    Regular bins as well as the staging areas are bins.

    Parameters
    ----------
    data : dict
        A dictionary containining the information about the object, as it is
        stored in the json file. Must have 'position', 'status' and
        'access_spots' keys.
    """
    def __init__(self, data):
        super().__init__(data)
        self.access_spots = [tuple(pos) for pos in data['access_spots']]

class Obstacle(WarehouseObject):
    """
    An obstacle.
    """
    pass

class StagingInArea(Bin):
    """
    The staging-in area.
    """
    pass


class StagingOutArea(Bin):
    """
    The staging-out area.
    The status variable doesn't represent the current contents of the bin,
    but rather the desired contents of the bin.
    """
    def put(self, slot, item):
        """
        Putting into the staging-out area works differently than putting into
        other containers: you can only put into a slot of the staging-out area
        if the item you're trying to put is requested in that slot.
        After putting, the slot will be empty.

        Parameters
        ----------
        slot : int
            The slot to put into.
        item : int
            The item to put into the slot.
        """
        assert self.status[slot] == item
        self.status[slot] = NO_ITEM

    def requires(self, slot, item):
        """
        Checks whether the item is required in the given slot.

        Parameters
        ----------
        slot : int
            The slot to check.
        item : int
            The item to check.

        Returns
        -------
        bool :
            True if the item is required in the given slot.
        """
        return self.status[slot] == item and item != NO_ITEM
