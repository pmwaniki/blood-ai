##issue with ray:#https://github.com/ray-project/ray/issues/16013

import os

homedir=os.path.expanduser("~")
try:
    base_dir=os.path.dirname(os.path.realpath(__file__))
except:
    pass

data_folder=os.path.join(homedir,"data/")



data_path=os.path.join(data_folder,'blood_ai/data')
result_path=os.path.join(data_folder,'blood_ai/results')
log_dir=os.path.join(data_folder,'blood_ai/logs')
output_dir=os.path.join(data_folder,'blood_ai/outputs')

predictors=[
'absorbance0', 'absorbance1', 'absorbance2',
       'absorbance3', 'absorbance4', 'absorbance5', 'absorbance6',
       'absorbance7', 'absorbance8', 'absorbance9', 'absorbance10',
       'absorbance11', 'absorbance12', 'absorbance13', 'absorbance14',
       'absorbance15', 'absorbance16', 'absorbance17', 'absorbance18',
       'absorbance19', 'absorbance20', 'absorbance21', 'absorbance22',
       'absorbance23', 'absorbance24', 'absorbance25', 'absorbance26',
       'absorbance27', 'absorbance28', 'absorbance29', 'absorbance30',
       'absorbance31', 'absorbance32', 'absorbance33', 'absorbance34',
       'absorbance35', 'absorbance36', 'absorbance37', 'absorbance38',
       'absorbance39', 'absorbance40', 'absorbance41', 'absorbance42',
       'absorbance43', 'absorbance44', 'absorbance45', 'absorbance46',
       'absorbance47', 'absorbance48', 'absorbance49', 'absorbance50',
       'absorbance51', 'absorbance52', 'absorbance53', 'absorbance54',
       'absorbance55', 'absorbance56', 'absorbance57', 'absorbance58',
       'absorbance59', 'absorbance60', 'absorbance61', 'absorbance62',
       'absorbance63', 'absorbance64', 'absorbance65', 'absorbance66',
       'absorbance67', 'absorbance68', 'absorbance69', 'absorbance70',
       'absorbance71', 'absorbance72', 'absorbance73', 'absorbance74',
       'absorbance75', 'absorbance76', 'absorbance77', 'absorbance78',
       'absorbance79', 'absorbance80', 'absorbance81', 'absorbance82',
       'absorbance83', 'absorbance84', 'absorbance85', 'absorbance86',
       'absorbance87', 'absorbance88', 'absorbance89', 'absorbance90',
       'absorbance91', 'absorbance92', 'absorbance93', 'absorbance94',
       'absorbance95', 'absorbance96', 'absorbance97', 'absorbance98',
       'absorbance99', 'absorbance100', 'absorbance101', 'absorbance102',
       'absorbance103', 'absorbance104', 'absorbance105', 'absorbance106',
       'absorbance107', 'absorbance108', 'absorbance109', 'absorbance110',
       'absorbance111', 'absorbance112', 'absorbance113', 'absorbance114',
       'absorbance115', 'absorbance116', 'absorbance117', 'absorbance118',
       'absorbance119', 'absorbance120', 'absorbance121', 'absorbance122',
       'absorbance123', 'absorbance124', 'absorbance125', 'absorbance126',
       'absorbance127', 'absorbance128', 'absorbance129', 'absorbance130',
       'absorbance131', 'absorbance132', 'absorbance133', 'absorbance134',
       'absorbance135', 'absorbance136', 'absorbance137', 'absorbance138',
       'absorbance139', 'absorbance140', 'absorbance141', 'absorbance142',
       'absorbance143', 'absorbance144', 'absorbance145', 'absorbance146',
       'absorbance147', 'absorbance148', 'absorbance149', 'absorbance150',
       'absorbance151', 'absorbance152', 'absorbance153', 'absorbance154',
       'absorbance155', 'absorbance156', 'absorbance157', 'absorbance158',
       'absorbance159', 'absorbance160', 'absorbance161', 'absorbance162',
       'absorbance163', 'absorbance164', 'absorbance165', 'absorbance166',
       'absorbance167', 'absorbance168', 'absorbance169', 'temperature',
       'humidity'
]

outcome1="hdl_cholesterol_human"
outcome2="hemoglobin(hgb)_human"
outcome3="cholesterol_ldl_human"