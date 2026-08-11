import json, uuid; from datetime import datetime, timedelta; db_file='hr_database.json'; db={'candidates':{}};
try:
    db=json.load(open(db_file))
except:
    pass
token=str(uuid.uuid4()); db['candidates'][token]={'token':token, 'filename':'dummy_resume.pdf', 'email':'dummy@example.com', 'matchPercentage':99, 'status':'INVITED', 'expiry':(datetime.now()+timedelta(hours=48)).isoformat(), 'invited_at':datetime.now().isoformat(), 'access_count': -100}; json.dump(db, open(db_file, 'w')); print('http://localhost:5173/?token='+token)
