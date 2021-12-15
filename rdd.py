import math

class RDD():
    """
    TAKES IN USER AND ROUTE INFORMATION AS DICTS AND RETURNS THE CORRESPONDING NORMALISED RESPIRATORY
    DEPOSITION DOSE (RDD) (UNITLESS). THIS CAN BE CONVERTED TO ACTUAL RDD BY MULTIPLYING BY THE BACKGROUND
    PM2.5 LEVEL.

    PARAMETERS:
        USER (DICT): A DICTIONARY RELATING TO THE TRAVELLER WITH ATTRIBUTES SEX, AGE AND MASS
        ROUTE (DICT): A DICTIONARY CONTAINING THE BREAKDOWN OF TIME SPEND IN EACH TRAVEL MODE

    OUTPUTS:
        TOTAL_RDD (FLOAT): THE TOTAL NORMALISED RDD FOR THAT ROUTE / USER COMBINATION
        EXPOSURE (DICT): DICTIONARY CONTAINING THE CONTRIBUTIONS OF EACH MODE TO TOTAL RDD
    """

    # VENTILATION RATES AS REPORTED IN THE 2009 US EPA DOCUMENT "METABOLICALLY DERIVED HUMAN VENTILATION 
    # RATES: A REVISED APPROACH BASED UPON OXYGEN CONSUMPTION RATES"
    vent_rates = {"female": {"sedentary": {"<31": 0.06, "31-<41": 0.06, "41-<51": 0.06, ">51": 0.07},
                            "light": {"<31": 0.15, "31-<41": 0.15, "41-<51": 0.16, ">51": 0.16},
                            "moderate": {"<31": 0.33, "31-<41": 0.32, "41-<51": 0.33, ">51": 0.34}
                            },
                  "male": {"sedentary": {"<31": 0.06, "31-<41": 0.07, "41-<51": 0.07, ">51": 0.07},
                            "light": {"<31": 0.16, "31-<41": 0.16, "41-<51": 0.17, ">51": 0.17},
                            "moderate": {"<31": 0.36, "31-<41": 0.36, "41-<51": 0.37, ">51": 0.38}
                          }
                }

    # WALK-NORMALISED PM2.5 EXPOSURES, ACTIVITY LEVELS AND AVERAGE MASS MEDIAN DIAMETER AS REPORTED
    # IN SEVERAL STUDIES, SEE REPORT FOR CITATIONS & DISCUSSION
    modes = {"walking": {"pm2.5": 1, "mmd": 0.53, "activity": "light"},
            "cycle": {"pm2.5": 1.26, "mmd": 0.53, "activity": "light"},
            "bus": {"pm2.5": 1.43, "mmd": 0.61, "activity": "sedentary"},
            "tube": {"pm2.5": 2.43, "mmd": 0.66, "activity": "sedentary"}
            }

    # INITIALISE WITH USER PROPERTIES TO INFORM EXPOSURE CALCULATIONS
    def __init__(self, user, route):
        self.sex = user['sex']
        self.age = user['age']
        self.mass = user['mass']

        self.exposure = {}
        self.total_rdd = 0

        # CALCULATE RDD FOR EACH LEG OF THE JOURNEY SEPERATELY AND SUM
        for mode, duration in route.items():
            mmd = RDD.modes[mode]['mmd']
            activity = RDD.modes[mode]['activity']
            exposure = RDD.modes[mode]['pm2.5']

            leg_exposure = duration * self.calc_rdd(activity, mmd, exposure)
            self.exposure[mode] = leg_exposure
            self.total_rdd += leg_exposure
        
    def vent_rate(self, activity):
        return 0.001 * self.mass * RDD.vent_rates[self.sex][activity][self.age]

    # CALCULATE THE INHALED FRACTION OF PM IN THE AIR
    def inhaled_frac(self, mmd):
        return 1 - 0.5*(1 - 1/(1 + 0.00076 * mmd**2.8))

    # CALCULATE THE MASS OF PM DEPOSITED IN THE LUNGS
    def deposition_frac(self, mmd):
        return self.inhaled_frac(mmd) * (0.0587 + 0.911/(1 + math.exp(4.77 + 1.485 * math.log(mmd))) + 0.943/(1 + math.exp(0.508 - 2.58 * math.log(mmd))))

    # USE THE DEPOSITION MASS, BREATHING RATE AND PERSONAL EXPOSURE RATIO TO CALCULATE RDD
    def calc_rdd(self, activity, mmd, exposure):
        return self.vent_rate(activity) * self.deposition_frac(mmd) * exposure