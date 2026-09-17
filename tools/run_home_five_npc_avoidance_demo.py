#!/usr/bin/env python3
"""Experiment-only five-NPC MuJoCo home navigation/avoidance recorder."""
from __future__ import annotations
import argparse, copy, json, math, subprocess, tempfile
from collections import deque
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2, mujoco, numpy as np
from stretch_mujoco.humanoid.navigation import OfficeNavigationMesh
from stretch_mujoco.npc.assets import NpcAssetManifest
from stretch_mujoco.npc.composition import compose_npc_scene
from stretch_mujoco.npc.locomotion import NavigationGeometryContract
from stretch_mujoco.npc.protocol import CommandStatus, NpcCommand, NpcCommandKind
from stretch_mujoco.npc.schema import NpcPopulation
from stretch_mujoco.npc.system import NpcSystem

ROOT=Path(__file__).resolve().parents[1]
CATALOG=ROOT/"stretch_mujoco/models/generated_scene_npc/active/active_catalog.json"
SCENE_ID="home_04_103997970_171031287"
ACTORS=tuple(f"npc_demo_{n:02d}" for n in range(1,6))

def inputs():
    c=json.loads(CATALOG.read_text()); e=c["scenes"][SCENE_ID]
    return c, CATALOG.parent/c["build_root"]/e["population"]
def nav(model,data):
    return OfficeNavigationMesh.from_model(model,data,floor_geom_name="hssd_floor_collision",resolution=.12,agent_radius=.18)
def routes_for(model,data):
    mesh=nav(model,data); cells=np.argwhere(mesh.component_labels==mesh.primary_component_id)
    pool=[mesh.cell_to_world((int(r),int(c))) for r,c in cells[::max(1,len(cells)//800)]]
    pts=[]
    while pool and len(pts)<16:
        p=min(pool,key=lambda q:(float(q[0]),float(q[1]))) if not pts else max(pool,key=lambda q:min(float(np.linalg.norm(q-x)) for x in pts))
        pts.append(np.asarray(p)); pool=[q for q in pool if np.linalg.norm(q-p)>=.95]
    choices=[]
    for i,a in enumerate(pts):
      for j,b in enumerate(pts):
       if i==j: continue
       try: path=mesh.plan(a,b)
       except Exception: continue
       if len(path)>3: choices.append((i,j,path))
    pair=None
    for n,a in enumerate(choices):
      for b in choices[n+1:]:
       if len({a[0],a[1],b[0],b[1]})<4: continue
       if any(np.linalg.norm(x[:2]-y[:2])<.42 for x in a[2][1:-1] for y in b[2][1:-1]): pair=(a,b); break
      if pair: break
    if not pair: raise RuntimeError("no_crossing_navigation_routes")
    chosen=[*pair]; used={pair[0][0],pair[0][1],pair[1][0],pair[1][1]}
    for item in choices:
      if item[0] not in used and item[1] not in used:
       chosen.append(item); used.update(item[:2])
       if len(chosen)==5: break
    if len(chosen)!=5: raise RuntimeError("not_enough_disjoint_navigation_routes")
    return [(pts[a],pts[b]) for a,b,_ in chosen]
def fixture(out):
    catalog,pop_path=inputs(); payload=json.loads(pop_path.read_text()); scene=(pop_path.parent/payload["scene"]).resolve()
    m=mujoco.MjModel.from_xml_path(str(scene)); d=mujoco.MjData(m); mujoco.mj_forward(m,d); routes=routes_for(m,d)
    tree=ET.parse(scene); world=tree.getroot().find("worldbody")
    compiler=tree.getroot().find("compiler")
    if compiler is not None and compiler.get("assetdir"):
        compiler.set("assetdir",str((scene.parent/compiler.get("assetdir")).resolve()))
    for include in tree.getroot().findall("include"):
        include.set("file",str((scene.parent/include.attrib["file"]).resolve()))
    for n,(start,target) in enumerate(routes,1):
      for role,p in (("spawn",start),("target",target)):
       ET.SubElement(world,"site",name=f"five_{n}_{role}",pos=f"{p[0]:.6f} {p[1]:.6f} .03",size=".025",rgba="0 0 0 0")
    base=out/"five_npc_home_base.xml"; ET.indent(tree,space="  "); tree.write(base,encoding="unicode")
    seeds=list(payload["npcs"].values()); payload["npcs"]={}
    for n,actor in enumerate(ACTORS,1):
      item=copy.deepcopy(seeds[(n-1)%len(seeds)]); item["agent_id"]=actor
      item["spawn"]={"location":"zone.home","site":f"five_{n}_spawn","yaw":0.0}; payload["npcs"][actor]=item
    payload["scene"]=str(base.resolve()); payload["asset_manifest"]=str((pop_path.parent/payload["asset_manifest"]).resolve()); payload.pop("trajectory_profile",None)
    if payload.get("appearance_catalog"):
        payload["appearance_catalog"]=str((pop_path.parent/payload["appearance_catalog"]).resolve())
    result=out/"five_npc.population.json"; result.write_text(json.dumps(payload,indent=2)+"\n")
    return result,routes,{"active_build_id":catalog["build_id"],"source_population":str(pop_path)}
class Video:
 def __init__(self,dest,model,w,h,fps):
  self.dest,self.w,self.h,self.fps=dest,w,h,fps; model.vis.global_.offwidth=max(model.vis.global_.offwidth,w);model.vis.global_.offheight=max(model.vis.global_.offheight,h)
  self.r=mujoco.Renderer(model,width=w,height=h);self.c=mujoco.MjvCamera();self.c.type=mujoco.mjtCamera.mjCAMERA_FREE;self.c.lookat[:]=model.stat.center;self.c.lookat[2]=.45;self.c.distance=max(8.5,model.stat.extent*1.85);self.c.azimuth=270;self.c.elevation=-86
  self.o=mujoco.MjvOption();mujoco.mjv_defaultOption(self.o);self.o.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER]=False
  self.tmp=Path(tempfile.NamedTemporaryFile(dir=dest.parent,suffix=".mp4",delete=False).name);self.v=cv2.VideoWriter(str(self.tmp),cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h));self.frames=0
  if not self.v.isOpened():raise RuntimeError("video_writer_unavailable")
 def frame(self,data,lines,step):
  self.r.update_scene(data,camera=self.c,scene_option=self.o);im=cv2.cvtColor(self.r.render(),cv2.COLOR_RGB2BGR);over=im.copy();cv2.rectangle(over,(0,self.h-112),(self.w,self.h),(8,14,22),-1);cv2.addWeighted(over,.84,im,.16,0,im)
  cv2.putText(im,f"5 NPC HOME | GLOBAL TOP-DOWN | MuJoCo step={step}",(16,28),cv2.FONT_HERSHEY_SIMPLEX,.55,(245,248,250),2,cv2.LINE_AA)
  for i,line in enumerate(list(lines)[-4:]):cv2.putText(im,line[:112],(16,self.h-84+i*22),cv2.FONT_HERSHEY_SIMPLEX,.43,(180,230,190),1,cv2.LINE_AA)
  self.v.write(im);self.frames+=1
 def close(self):
  self.v.release();self.r.close();subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(self.tmp),"-c:v","libx264","-pix_fmt","yuv420p","-movflags","+faststart","-an",str(self.dest)],check=True);self.tmp.unlink(missing_ok=True)
  return {"path":str(self.dest),"frames":self.frames,"fps":self.fps,"width":self.w,"height":self.h,"camera":"global_top_down"}
def run(out,fps,w,h):
 out.mkdir(parents=True,exist_ok=True); p,routes,source=fixture(out); comp=compose_npc_scene(p,out/"five_npc_composed.xml",reuse=False); pop=NpcPopulation.from_json(p); manifest=NpcAssetManifest.from_json(Path(pop.asset_manifest))
 m=mujoco.MjModel.from_xml_path(str(comp.scene_path));d=mujoco.MjData(m);mujoco.mj_forward(m,d);system=NpcSystem.from_population(m,pop,manifest,simulation_seed=20260917)
 # Deliberately inflate the runtime footprint beyond the visual torso radius
 # for a legible no-pass-through multi-NPC presentation.
 contract=NavigationGeometryContract("hssd_floor_collision",.20,0,.06,("base_link",))
 for controller in system.controllers.values():controller.locomotion.configure_navigation(contract);controller.locomotion.dynamic_obstacles=True
 video=Video(out/"five_npc_home_dynamic_avoidance.mp4",m,w,h,fps);events=[];lines=deque(["all 5 MOVE_TO accepted; dynamic obstacles enabled"],maxlen=4);receipts={};contacts=[];replans={a:0 for a in ACTORS};minimum=math.inf;overlap=0;commands=[]
 for n,a in enumerate(ACTORS,1):
  command=NpcCommand(f"five-npc:{a}:move",0,a,NpcCommandKind.MOVE_TO,{"site":f"five_{n}_target","speed":.82,"max_replans":12,"progress_timeout":5.0},0)
  accepted=system.submit(command)
  if accepted.status is CommandStatus.FAILED:raise RuntimeError(f"command_rejected:{a}:{accepted.reason}")
  commands.append(command)
 t=0
 try:
  for step in range(1,1401):
   t+=.05;system.step(m,d,t);mujoco.mj_forward(m,d);states=system.states(d,t);moving=[a for a in ACTORS if states[a].active_command_id];overlap+=len(moving)>=2
   poses={a:d.mocap_pos[system.controllers[a].binding.mocap_id,:2].copy() for a in ACTORS};minimum=min(minimum,*(float(np.linalg.norm(poses[a]-poses[b])) for i,a in enumerate(ACTORS) for b in ACTORS[i+1:]))
   for a,c in system.controllers.items():
    if c.locomotion.route_revision>replans[a]:
     e={"type":"dynamic_avoidance_replan","step":step,"time_s":t,"actor":a,"route_revision":c.locomotion.route_revision};events.append(e);lines.append(f"AVOID {a}: dynamic replan #{c.locomotion.route_revision}");replans[a]=c.locomotion.route_revision
   for k in range(d.ncon):
    x=d.contact[k];bs={int(m.geom_bodyid[x.geom1]),int(m.geom_bodyid[x.geom2])};owners=[a for a in ACTORS if system.controllers[a].binding.body_id in bs]
    if len(owners)==2:contacts.append({"step":step,"pair":sorted(owners),"distance":float(x.dist)})
   for c in commands:
    x=states[c.npc_id].last_receipt
    if x and x.command_id==c.command_id and x.status.terminal:receipts[c.npc_id]=x.to_dict()
   if step%2==0:lines.append(f"moving={len(moving)} min separation={minimum:.2f}m contacts={len(contacts)}");video.frame(d,lines,step)
   if len(receipts)==5:break
 finally: meta=video.close()
 events += [{"type":"terminal_receipt","actor":a,"receipt":x} for a,x in sorted(receipts.items())]
 (out/"events.jsonl").write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in events))
 avoidance=[x for x in events if x["type"]=="dynamic_avoidance_replan"]
 succeeded=[a for a,x in receipts.items() if x["status"]=="succeeded"]
 report={"schema":"home_five_npc_dynamic_avoidance_demo/v1","scene_id":SCENE_ID,"source":source,"actors":list(ACTORS),"routes":[{"actor":a,"start_xy":list(map(float,s)),"target_xy":list(map(float,g))} for a,(s,g) in zip(ACTORS,routes)],"navigation":{"dynamic_obstacles":True,"footprint_radius_m":.20,"simultaneous_navigation_steps":overlap,"avoidance_replans":avoidance,"min_center_distance_m":minimum,"npc_npc_contacts":contacts,"collision_free":not contacts},"terminal_receipts":receipts,"video":meta,"passed":len(receipts)==5 and len(succeeded)>=2 and overlap>0 and minimum>=.32 and not contacts and bool(avoidance),"limitations":["experiment-only temporary population","some traffic routes may terminate fail-closed as route_unavailable","no Stretch robot command was issued"]}
 (out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");(out/"README.md").write_text("# Five NPC dynamic-avoidance home demo\n\nThis experimental output renders a temporary five-NPC population over existing home_04. All five MOVE_TO commands are submitted together using the existing dynamic-mocap-obstacle navigation mesh. events.jsonl contains observed replans and receipts; report.json contains MuJoCo contact scans and minimum separation. No active catalog, source home scene, production population, or Stretch command is modified. The subtitle panel shows actor/action/avoidance timeline.\n")
 return report
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,default=ROOT/"aaa_workspace/experiments/home_five_npc_dynamic_avoidance");ap.add_argument("--fps",type=int,default=10);ap.add_argument("--width",type=int,default=960);ap.add_argument("--height",type=int,default=640);a=ap.parse_args();r=run(a.output_dir.resolve(),a.fps,a.width,a.height);print(json.dumps({"output_dir":str(a.output_dir.resolve()),"passed":r["passed"],"video":r["video"]["path"]}));return 0 if r["passed"] else 1
if __name__=="__main__":raise SystemExit(main())
