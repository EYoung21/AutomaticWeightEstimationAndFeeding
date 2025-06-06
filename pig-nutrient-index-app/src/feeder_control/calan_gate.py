class CalanGateController:
    def __init__(self, min_feed=1.0, max_feed=5.0):
        """
        min_feed: Minimum feed amount (kg)
        max_feed: Maximum feed amount (kg)
        """
        # Initialize the Calan gate controller
        self.gate_status = {}  # Dictionary to hold the status of each gate
        self.min_feed = min_feed
        self.max_feed = max_feed

    def open_gate(self, pig_id):
        # Open the gate for the specified pig
        if pig_id not in self.gate_status:
            self.gate_status[pig_id] = 'closed'  # Initialize gate status if not present
        self.gate_status[pig_id] = 'open'
        print(f"Gate opened for pig {pig_id}")

    def close_gate(self, pig_id):
        # Close the gate for the specified pig
        if pig_id not in self.gate_status:
            self.gate_status[pig_id] = 'open'  # Initialize gate status if not present
        self.gate_status[pig_id] = 'closed'
        print(f"Gate closed for pig {pig_id}")

    def control_gate(self, pig_id, nutrient_index):
        # Control the gate based on the nutrient index
        if nutrient_index < 0:
            print(f"Nutrient index for pig {pig_id} is too low. Medical intervention needed.")
            self.close_gate(pig_id)  # Close gate if nutrient index is too low
        elif nutrient_index < 50:
            print(f"Nutrient index for pig {pig_id} is low. Consider increasing feed.")
            self.open_gate(pig_id)  # Open gate for feeding
        elif nutrient_index >= 50:
            print(f"Nutrient index for pig {pig_id} is sufficient. Gate remains closed.")
            self.close_gate(pig_id)  # Keep gate closed if nutrient index is sufficient

    def get_gate_status(self, pig_id):
        # Return the current status of the gate for the specified pig
        return self.gate_status.get(pig_id, 'closed')  # Default to 'closed' if not found

    def get_feed_amount(self, nutrient_index):
        """
        Determines feed amount based on nutrient index.
        Args:
            nutrient_index (int): Nutrient index (0-100)
        Returns:
            float: Feed amount in kg
        """
        # Linear scaling: lower index = more feed, higher index = less feed
        feed = self.max_feed - ((nutrient_index / 100) * (self.max_feed - self.min_feed))
        return round(feed, 2)

    def control_feeder(self, pig_id, feed_amount):
        """
        Simulates sending a command to the feeder hardware.
        Args:
            pig_id (str): Unique pig identifier
            feed_amount (float): Feed amount in kg
        """
        # In real implementation, send command to hardware here
        print(f"Feeder for pig {pig_id}: Dispense {feed_amount} kg of feed.")