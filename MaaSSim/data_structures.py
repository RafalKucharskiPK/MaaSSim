from dotmap import DotMap
import pandas as pd


# general data structure, dictionary of DataFrames with predefined columns (minimal definition)
structures = DotMap()
structures.passengers = pd.DataFrame(columns=['id', 'pos', 'event', 'platforms']).set_index('id')
structures.vehicles = pd.DataFrame(columns=['id', 'pos', 'event',
                                            'shift_start', 'shift_end', 'platform','expected_income']).set_index('id')
structures.platforms = pd.DataFrame(columns=['id', 'fare', 'name', 'batch_time']).set_index('id')
structures.requests = pd.DataFrame(columns=['pax', 'pax_id', 'origin', 'destination',
                                            'treq', 'tdep', 'ttrav', 'tarr', 'tdrop',
                                            'shareable', 'schedule_id']).set_index('pax')

structures.schedule = pd.DataFrame(columns=['id', 'node', 'time', 'req_id', 'od']).set_index('id')