import requests
from datetime import datetime, timedelta
from decouple import config

from rdd import RDD

class RouteFactory():
    """
    TAKES IN START AND END POINTS TO GENERATE MULTIPLE ROUTES BETWEEN THE TWO USING TFL'S JOURNEY API.

    PARAMETERS:
        START (STR): THE POSTCODE OR STATION ID OF THE ORIGIN POINT
        END (STR): THE POSTCODE OR STATION ID OF THE END POINT

    KEY METHODS:
        GET_ROUTES(): RETURNS A LIST OF ROUTE OBJECTS
    """

    # DEFINE KEYS USED TO ACCESS TFL API
    appID = config('appID')
    appKey = config('appKey')
    
    # SETUP JOURNEY WITH START AND END POINTS, A START TIME AND ANY ARRIVAL DEADLINE
    def __init__(self, start=None, end=None, bg_exposure=1, time=datetime.now(), deadline=None):
        if start==None: start = 'W6 0XP'
        if end==None: end = 'SW7 2BX'

        self.start = start
        self.end = end
        self.time = time.strftime('%H%M')
        self.bg_exposure = bg_exposure

        if deadline: self.deadline = datetime.strptime(deadline, '%H%M')
        else: self.deadline = deadline

    # FETCH SUGGESTED MULTI-MODAL ROUTES FROM THE TFL API
    def get_routes(self):
        payload = {'time': self.time, 'timels': 'departing', 'useMultiModalCall': True}
        r = requests.get(f"https://api.tfl.gov.uk/journey/journeyresults/{self.start}/to/{self.end}&app_id={RouteFactory.appID}&app_key={RouteFactory.appKey}", params=payload)

        if r.status_code != 200:
            raise RuntimeError('The API request failed.')
        else:
            return self.parse_routes(r.json())

    # BREAK DOWN THE API RESPONSE INTO INDIVIDUAL ROUTES, AND CREATE ROUTE OBJECTS FOR 
    # EACH ROUTE PROVIDED WITH DURATION, ARRIVAL TIME, FARE AND MODES ATTRIBUTES
    def parse_routes(self,r):
        if len(r['journeys']) == 0:
            raise RuntimeError('No paths were found.')

        routes = {}

        for i, route in enumerate(r['journeys']):
            duration = route['duration']
            arrive_time = datetime.fromisoformat(route['arrivalDateTime'])

            if (self.deadline) and (arrive_time.time() > self.deadline.time()):
                continue
            
            try:
                charge_info = route['fare']['fares']
                fare = 0

                # IMPLEMENT 0.66 SCALING FOR OFF-PEAK JOURNEYS WITH RAILCARD
                for paid_leg in charge_info:
                    fare_type = paid_leg.get('chargeLevel', None)
                    if fare_type == 'Off Peak':
                        fare += paid_leg['cost'] * 0.66
                    else:
                        fare += paid_leg['cost']
                    
            except KeyError:
                fare = 0

            modes = []
            legs = []
            for leg in route['legs']:
                modes.append([leg['mode']['name'], leg['duration']])
                legs.append([leg['departurePoint']['commonName'], leg['instruction']['detailed'], leg['arrivalPoint']['commonName']])

            # CREATE ROUTE OBJECT AND STORE ITS REFERENCE IN A LIST OF ROUTES
            x = Route(duration, arrive_time, fare, modes, legs, self.bg_exposure)
            routes[i] = x

        # RETURN THE LIST OF FEASIBLE ROUTES
        return routes

class Route():
    """
    ROUTE OBJECT CONTAINING THE DETAIL AND ATTRIBUTES OF A ROUTE FROM A TO B. SHOULD ONLY BE CALLED
    THROUGH ROUTEFACTORY.

    KEY METHODS:
        FOLLOW(): RETURNS HUMAN-READABLE INSTRUCTIONS TO FOLLOW THE ROUTE
        CALC_RDD(USER): RETURNS THE NORMALISED RDD ASSOCIATED WITH THE ROUTE FOR A GIVEN USER
    """

    def __init__(self, duration, arrive_time, fare, modes, legs, bg_exposure):        
        self.duration = duration
        self.arrive_time = arrive_time
        self.depart_time = arrive_time - timedelta(minutes=duration)

        self.fare = fare
        self.modes = modes

        self.legs = legs

        self.txt_mode = self.route_type()
        self.rdd = self.calc_rdd(bg_exposure)

    # REPR METHOD TO ALLOW ROUTE DETAILS TO BE PRINTED
    #def __repr__(self):
    #    return f"{str(self.duration)} minute {self.route_type()} journey costing £{self.fare/100:.2f} arriving at {str(self.arrive_time.strftime('%X'))}."

    # IDENTIFY THE DOMINANT TRANSPORT MODE(S) IN THE ROUTE
    def route_type(self):
        if len(self.modes) == 1:
            return self.modes[0][0]

        else:
            modes = ''
            for mode in self.modes[1:-1]:
                if mode != self.modes[1]:
                    modes += ' and '
                modes += mode[0]
            return modes

    # PROVIDE HUMAN-READABLE DIRECTIONS FOR EACH STEP OF THE ROUTE
    def follow(self):
        print(f"Leave {self.legs[0][0]} by {str(self.depart_time.strftime('%X'))}.")
        for leg in self.legs:
            print(f"{leg[1]} until you arrive at {leg[2]}.")

    # CALCULATE THE RDD OF THE ROUTE FOR A GIVEN USER
    def calc_rdd(self, background_level, user={"sex": 'male', "age": '31-<41', "mass": 70}):
        journey = {}
        for leg, duration in self.modes:
            if journey.get(leg) != None:
                journey[leg] = journey[leg] + duration
            else:
                journey[leg] = duration
        exposure = RDD(user, journey)
        return exposure.total_rdd * background_level