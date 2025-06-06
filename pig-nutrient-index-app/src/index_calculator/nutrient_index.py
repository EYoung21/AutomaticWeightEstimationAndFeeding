class NutrientIndexCalculator:
    def __init__(self, min_weight=30, max_weight=120):
        """
        Initializes the NutrientIndexCalculator with the minimum and maximum healthy weights
        for the pig.
        
        :param min_weight: Minimum healthy weight (kg)
        :param max_weight: Maximum healthy weight (kg)
        """
        self.min_weight = min_weight
        self.max_weight = max_weight

    def calculate_index(self, weight):
        """
        Calculates the nutrient index for a given weight.
        
        :param weight: Estimated pig weight in kg
        :return: Nutrient index (0-100)
        """
        if weight <= self.min_weight:
            return 0
        elif weight >= self.max_weight:
            return 100
        else:
            index = ((weight - self.min_weight) / (self.max_weight - self.min_weight)) * 100
            return int(index)