import math

class DistCalculation:
    @staticmethod
    def haversine(Latitude1, Longitude1, Latitude2, Longitude2):
        from math import sqrt
        from math import sin, cos, sqrt, atan2, radians
        
        distance = 0
        # Approximate radius of earth in km
        R = 6373.0
        lat1 = radians(Latitude1)
        lon1 = radians(Longitude1)
        lat2 = radians(Latitude2)
        lon2 = radians(Longitude2)
        
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        try:
            a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            distance = R * c
        except Exception as e:
            return 0
            
        finally:
            return distance

# Example usage:
# dist = DistCalculation.haversine(52.2296756, 21.0122287, 41.8919300, 12.5113300)
# print(f"Distance: {dist} km")

