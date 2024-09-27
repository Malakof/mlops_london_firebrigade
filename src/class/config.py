import os

# Définition des chemins
chemin_racine = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

chemin_src = chemin_racine +'\Scr\\'
chemin_data = chemin_racine +'\dataset\\'
chemin_data_ref = chemin_racine +'\dataset\\ref\\'
chemin_model = chemin_racine +'\model\\'

#définition des urls
url_incident="https://data.london.gov.uk/download/london-fire-brigade-incident-records/f5066d66-c7a3-415f-9629-026fbda61822/LFB%20Incident%20data%20from%202018%20onwards.csv.xlsx"
url_mobilisation="https://data.london.gov.uk/download/london-fire-brigade-mobilisation-records/3ff29fb5-3935-41b2-89f1-38571059237e/LFB%20Mobilisation%20data%202021%20-%202024.xlsx"


# Définition des nom de fichier
fichier_incident = "incident_data.csv"
fichier_mobilisation = "mobilisation_data.csv"
fichier_stations = "stations.csv"
fichier_calendrier = "calendrier.csv"
fichier_vehicle = "vehicle.csv"
fichier_global = "global_data.csv"
fichier_model = "linear_regression_model.pkl"

# définition de la période d'analyse
years = [2022,2023]

#définition des dimensions

incident_expected_columns = ['IncidentNumber','DateOfCall','CalYear','TimeOfCall','HourOfCall','IncidentGroup','StopCodeDescription',
                            'SpecialServiceType','PropertyCategory','PropertyType','AddressQualifier','Postcode_full','Postcode_district',
                            'UPRN','USRN','IncGeo_BoroughCode','IncGeo_BoroughName','ProperCase','IncGeo_WardCode','IncGeo_WardName','IncGeo_WardNameNew'
                            ,'Easting_m','Northing_m','Easting_rounded','Northing_rounded','Latitude','Longitude','FRS','IncidentStationGround',
                            'FirstPumpArriving_AttendanceTime','FirstPumpArriving_DeployedFromStation','SecondPumpArriving_AttendanceTime',
                            'SecondPumpArriving_DeployedFromStation','NumStationsWithPumpsAttending','NumPumpsAttending','PumpCount','PumpMinutesRounded',
                            'Notional Cost (£)','NumCalls']

mobilisation_expected_columns = ['IncidentNumber','CalYear','HourOfCall','ResourceMobilisationId','Resource_Code','PerformanceReporting',
                                'DateAndTimeMobilised','DateAndTimeMobile','DateAndTimeArrived','TurnoutTimeSeconds','TravelTimeSeconds',
                                'AttendanceTimeSeconds','DateAndTimeLeft','DateAndTimeReturned','DeployedFromStation_Code','DeployedFromStation_Name',
                                'DeployedFromLocation','PumpOrder','PlusCode_Code','PlusCode_Description','DelayCodeId','DelayCode_Description']

CalYear_incident = 'CalYear'
CalYear_mobilisation = 'CalYear'

BandWidth_speed_min = 5
BandWidth_speed_max = 35
BandWidth_AttendanceTimeSeconds_min = 193
BandWidth_AttendanceTimeSeconds_max = 539




