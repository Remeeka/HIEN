import numpy as np, pandas as pd
from datetime import datetime, timedelta
rng = np.random.default_rng(42)
START = datetime(2026,7,1); DAYS = 90
WH = (0.0, 0.0)  # central warehouse at origin, coords in km on a 60x60 km synthetic metro grid

# ---------- locations ----------
nL = 40
zones = ['North','South','East','West','Central']
loc = pd.DataFrame({
 'location_id':[f'LOC{i:03d}' for i in range(1,nL+1)],
 'location_type':['Store']*25+['Access Point']*15,
})
loc['zone']=rng.choice(zones,nL)
loc['x_km']=rng.uniform(-30,30,nL).round(2); loc['y_km']=rng.uniform(-30,30,nL).round(2)
loc['storage_capacity_units']=np.where(loc.location_type=='Store',rng.integers(150,400,nL),rng.integers(30,120,nL))
loc['verification_slots_per_day']=np.where(loc.location_type=='Store',rng.integers(25,60,nL),rng.integers(8,25,nL))
loc['avg_daily_footfall']=np.where(loc.location_type=='Store',rng.integers(300,1500,nL),rng.integers(40,300,nL))
loc['dist_to_warehouse_km']=np.hypot(loc.x_km-WH[0],loc.y_km-WH[1]).round(2)
loc['hub_enabled']=rng.random(nL)>0.1
loc.to_csv('locations.csv',index=False)

# ---------- products ----------
cats = {  # name: (price_lo, price_hi, weight_kg, resale_ok_prob, hygiene, fragile, daily_demand_weight)
 'Apparel':(15,120,0.4,0.80,0,0,5),'Footwear':(25,150,0.9,0.78,0,0,3),'Electronics':(40,900,1.2,0.50,0,1,2.5),
 'Home & Kitchen':(10,200,2.0,0.70,0,1,3),'Beauty':(8,80,0.2,0.30,1,0,2.5),'Toys':(8,90,0.7,0.82,0,0,2),
 'Books':(6,40,0.4,0.92,0,0,1.5),'Sports':(15,200,1.5,0.75,0,0,1.5),'Accessories':(10,150,0.3,0.80,0,0,2)}
rows=[];n=0
for c,(lo,hi,w,p,h,f,dw) in cats.items():
    for _ in range(14):
        n+=1; rows.append(dict(sku=f'SKU{n:04d}',category=c,unit_price=round(float(np.exp(rng.uniform(np.log(lo),np.log(hi)))),2),
          weight_kg=round(w*rng.uniform(.6,1.6),2),hygiene_sensitive=h,fragile=f,base_resale_prob=p,demand_weight=dw))
prod=pd.DataFrame(rows); prod.to_csv('products.csv',index=False)
P=prod.set_index('sku')

# ---------- returns ----------
nR=7000
p_w = prod.demand_weight/prod.demand_weight.sum()
r_sku=rng.choice(prod.sku,nR,p=p_w)
r_loc=rng.choice(loc.location_id,nR,p=(loc.avg_daily_footfall/loc.avg_daily_footfall.sum()))
r_time=[START+timedelta(minutes=int(m)) for m in rng.integers(0,DAYS*24*60-60*24*7,nR)]  # leave a week tail for sales
reasons=['Size/fit','Changed mind','Defective','Wrong item','Not as described','Damaged in transit','Late delivery','Better price found']
reason_p=[.25,.22,.10,.06,.14,.05,.06,.12]
reason=rng.choice(reasons,nR,p=reason_p)
R=pd.DataFrame({'return_id':[f'RET{i:06d}' for i in range(1,nR+1)],'return_timestamp':r_time,'sku':r_sku,'return_location_id':r_loc,'return_reason':reason})
R['category']=R.sku.map(P.category); R['unit_price']=R.sku.map(P.unit_price); R['weight_kg']=R.sku.map(P.weight_kg)
R['hygiene_sensitive']=R.sku.map(P.hygiene_sensitive); R['fragile']=R.sku.map(P.fragile)
R['days_since_purchase']=rng.integers(1,31,nR)
R['has_original_packaging']=rng.random(nR)<np.where(R.return_reason.isin(['Defective','Damaged in transit']),.6,.88)
R['is_fraud_swap']=rng.random(nR)<0.025
# true condition
base=R.sku.map(P.base_resale_prob).values
bad_reason=R.return_reason.isin(['Defective','Damaged in transit']).values
p_good=np.clip(base-0.45*bad_reason-0.15*(~R.has_original_packaging.values)-0.6*R.is_fraud_swap.values,0.02,0.97)
good=rng.random(nR)<p_good
tc=np.where(good,np.where(rng.random(nR)<.55,'New/Sealed','Like New'),np.where(rng.random(nR)<.5,'Minor Wear','Damaged'))
R['true_condition']=tc
# AI grade: noisy version of truth
order=['Damaged','Minor Wear','Like New','New/Sealed']
idx={c:i for i,c in enumerate(order)}
ti=np.array([idx[c] for c in tc])
noise_sd=np.where(R.category.isin(['Electronics']),0.75,np.where(R.category.isin(['Apparel','Books','Toys','Footwear']),0.35,0.5))
gi=np.clip(np.rint(ti+rng.normal(0,noise_sd)),0,3).astype(int)
R['ai_condition_grade']=[order[i] for i in gi]
R['ai_confidence']=np.clip(rng.normal(0.88,0.08,nR)-0.2*(gi!=ti),0.4,0.99).round(3)
R['ai_fraud_risk_score']=np.clip(np.where(R.is_fraud_swap,rng.normal(.6,.2,nR),rng.normal(.1,.08,nR)),0,1).round(3)
# local demand score at return location (0-1), later refined by sim
loc_foot=loc.set_index('location_id').avg_daily_footfall
R['local_demand_score']=np.clip((R.return_location_id.map(loc_foot)/1500)*0.5+R.sku.map(P.demand_weight)/5*0.4+rng.normal(0,.08,nR),0,1).round(3)
elig=(gi>=2)&(R.ai_confidence>=0.75)&(R.ai_fraud_risk_score<0.35)&(R.hygiene_sensitive==0)&R.return_location_id.map(loc.set_index('location_id').hub_enabled)
R['hien_eligible']=elig
R['routing_decision']=np.where(elig,'Local Pool',np.where(gi>=1,'Return to Warehouse','Liquidate/Dispose'))
R['grade_correct']=(gi==ti)
R['misgraded_as_resalable']=(gi>=2)&(ti<2)

# ---------- demand / orders ----------
nO=24000
o_sku=rng.choice(prod.sku,nO,p=p_w)
o_t=np.sort(rng.integers(0,DAYS*24*60,nO))
# customers cluster around locations
o_loc=rng.choice(loc.location_id,nO,p=(loc.avg_daily_footfall/loc.avg_daily_footfall.sum()))
L=loc.set_index('location_id')
cx=L.loc[o_loc,'x_km'].values+rng.normal(0,2.5,nO); cy=L.loc[o_loc,'y_km'].values+rng.normal(0,2.5,nO)
O=pd.DataFrame({'order_id':[f'ORD{i:06d}' for i in range(1,nO+1)],'order_timestamp':[START+timedelta(minutes=int(m)) for m in o_t],
 'sku':o_sku,'customer_x_km':cx.round(2),'customer_y_km':cy.round(2),'nearest_location_id':o_loc})
O['unit_price']=O.sku.map(P.unit_price); O['weight_kg']=O.sku.map(P.weight_kg); O['category']=O.sku.map(P.category)
O['promised_delivery']=rng.choice(['Standard','Express'],nO,p=[.7,.3])

# ---------- simulate matching ----------
pool=R[R.hien_eligible].sort_values('return_timestamp')
pool_items=[dict(rid=r.return_id,sku=r.sku,loc=r.return_location_id,t=r.return_timestamp,x=L.loc[r.return_location_id,'x_km'],y=L.loc[r.return_location_id,'y_km']) for r in pool.itertuples()]
pi=0; avail={}  # sku -> list of items
MAX_DWELL=timedelta(days=14); RADIUS=10.0
sold={}; fulfil=[];
for o in O.itertuples():
    while pi<len(pool_items) and pool_items[pi]['t']<=o.order_timestamp:
        it=pool_items[pi]; avail.setdefault(it['sku'],[]).append(it); pi+=1
    lst=avail.get(o.sku,[])
    lst[:]=[i for i in lst if o.order_timestamp-i['t']<=MAX_DWELL]
    best=None;bd=1e9
    for it in lst:
        d=np.hypot(it['x']-o.customer_x_km,it['y']-o.customer_y_km)
        if d<bd: bd=d;best=it
    if best is not None and bd<=RADIUS:
        lst.remove(best); sold[best['rid']]=(o.order_id,o.order_timestamp,best['loc']); fulfil.append((o.order_id,'HIEN Local',best['loc'],round(bd,2),best['rid']))
    else:
        d=np.hypot(o.customer_x_km-WH[0],o.customer_y_km-WH[1]); fulfil.append((o.order_id,'Central Warehouse','WH001',round(d,2),None))
F=pd.DataFrame(fulfil,columns=['order_id','fulfilled_from_type','fulfilled_from_id','delivery_distance_km','source_return_id'])
O=O.merge(F,on='order_id')
O['delivery_hours']=np.where(O.fulfilled_from_type=='HIEN Local',np.clip(rng.normal(3.5,1.5,nO)+O.delivery_distance_km*.15,1,12),
                             np.clip(rng.normal(30,8,nO)+O.delivery_distance_km*.4,12,72)).round(1)
O['delivery_cost']=np.where(O.fulfilled_from_type=='HIEN Local',1.2+0.35*O.delivery_distance_km,3.5+0.12*O.delivery_distance_km).round(2)
O['co2_kg']=np.where(O.fulfilled_from_type=='HIEN Local',0.12*O.delivery_distance_km,0.12*O.delivery_distance_km+0.9).round(3)
O.to_csv('orders.csv',index=False)

# ---------- finalize returns with baseline vs HIEN cost ----------
R['sold_via_order_id']=R.return_id.map(lambda r: sold.get(r,(None,))[0])
R['resold_timestamp']=R.return_id.map(lambda r: sold.get(r,(None,None))[1])
R['dwell_days']=((pd.to_datetime(R.resold_timestamp)-R.return_timestamp).dt.total_seconds()/86400).round(2)
d_wh=R.return_location_id.map(L.dist_to_warehouse_km)
R['baseline_return_cost']=(2.0+0.5*d_wh+0.8*R.weight_kg+3.0).round(2)       # carrier + warehouse inspect/restock
R['baseline_return_distance_km']=d_wh
R['hien_return_cost']=np.where(R.hien_eligible,(0.8+0.15*R.weight_kg+0.6).round(2),R.baseline_return_cost)  # local handling only
R['hien_return_distance_km']=np.where(R.hien_eligible,0.0,d_wh)
R['final_status']=np.where(~R.hien_eligible,R.routing_decision,np.where(R.sold_via_order_id.notna(),'Resold Locally','Expired - Sent to Warehouse'))
R.to_csv('returns.csv',index=False)

# ---------- local inventory snapshots (daily, per location x category) ----------
snap=[]
for d in range(0,DAYS,1):
    day=START+timedelta(days=d)
    sub=R[R.hien_eligible&(R.return_timestamp<=day+timedelta(days=1))]
    soldd=pd.to_datetime(sub.resold_timestamp)
    on=sub[(soldd.isna()|(soldd>day+timedelta(days=1)))&((day+timedelta(days=1)-sub.return_timestamp)<=MAX_DWELL)]
    g=on.groupby(['return_location_id','category']).size().reset_index(name='units_on_hand')
    g.insert(0,'date',day.date()); snap.append(g)
S=pd.concat(snap).rename(columns={'return_location_id':'location_id'})
S['capacity_utilisation']=(S.units_on_hand/S.location_id.map(L.storage_capacity_units)).round(3)
S.to_csv('inventory_snapshots.csv',index=False)

# ---------- weekly demand forecast vs actual ----------
O['week']=((pd.to_datetime(O.order_timestamp)-START).dt.days//7)
act=O.groupby(['nearest_location_id','category','week']).size().reset_index(name='actual_demand_units')
act['forecast_demand_units']=np.clip(np.rint(act.actual_demand_units*rng.normal(1.0,0.22,len(act))+rng.normal(0,.6,len(act))),0,None).astype(int)
act=act.rename(columns={'nearest_location_id':'location_id'}); act.to_csv('demand_forecast.csv',index=False)
O.drop(columns='week').to_csv('orders.csv',index=False)

# ---------- summary ----------
e=R[R.hien_eligible]
print('returns',len(R),'eligible%',round(R.hien_eligible.mean()*100,1),'resold locally % of eligible',round((e.final_status=="Resold Locally").mean()*100,1))
print('misgraded resalable among eligible %',round(e.misgraded_as_resalable.mean()*100,2))
print('baseline cost',R.baseline_return_cost.sum().round(0),'hien',R.hien_return_cost.sum().round(0),'saving%',round((1-R.hien_return_cost.sum()/R.baseline_return_cost.sum())*100,1))
print('distance saving%',round((1-R.hien_return_distance_km.sum()/R.baseline_return_distance_km.sum())*100,1))
print('local orders %',round((O.fulfilled_from_type=="HIEN Local").mean()*100,1),'median hrs',O.groupby('fulfilled_from_type').delivery_hours.median().to_dict())
print(R.final_status.value_counts())
