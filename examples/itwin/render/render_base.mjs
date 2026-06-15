// Render the Baytown plant ONCE to a base PNG, and dump the exact camera so a
// separate overlay pass (Python/cv2) can project Retina world points onto the
// same pixels. Pure Node — reads scene.bin (no backend needed here).
import fs from "fs";
import zlib from "zlib";

const buf = fs.readFileSync("scene.bin");
const nVerts = buf.readUInt32LE(0), nTris = buf.readUInt32LE(4);
const pos = new Float32Array(buf.buffer, buf.byteOffset + 8, nVerts * 3);
const idx = new Uint32Array(buf.buffer, buf.byteOffset + 8 + nVerts * 3 * 4, nTris * 3);

const lo = [Infinity,Infinity,Infinity], hi = [-Infinity,-Infinity,-Infinity];
for (let i=0;i<nVerts;i++) for(let k=0;k<3;k++){const v=pos[i*3+k]; if(v<lo[k])lo[k]=v; if(v>hi[k])hi[k]=v;}
const ctr=[(lo[0]+hi[0])/2,(lo[1]+hi[1])/2,(lo[2]+hi[2])/2];
const span=Math.max(hi[0]-lo[0],hi[1]-lo[1],hi[2]-lo[2]);

const az=(-35)*Math.PI/180, el=(28)*Math.PI/180;
const dir=[Math.cos(el)*Math.cos(az),Math.cos(el)*Math.sin(az),Math.sin(el)];
const fwd=[-dir[0],-dir[1],-dir[2]];
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const norm=(a)=>{const l=Math.hypot(...a)||1;return [a[0]/l,a[1]/l,a[2]/l];};
let up=[0,0,1]; let right=norm(cross(fwd,up)); up=norm(cross(right,fwd));
const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
ctr[2]=13;  // frame lower: emphasise the ground slab / activity, not the empty sky above the column
const W=1280,H=860,margin=1.06,scale=(Math.min(W,H)*margin)/span;
const project=(p)=>{const rel=[p[0]-ctr[0],p[1]-ctr[1],p[2]-ctr[2]];return [W/2+dot(rel,right)*scale,H/2-dot(rel,up)*scale,dot(rel,fwd)];};

const color=new Uint8Array(W*H*3), zbuf=new Float32Array(W*H).fill(Infinity);
for(let y=0;y<H;y++){const t=y/H,r=235-30*t,g=238-28*t,b=244-22*t;for(let x=0;x<W;x++){const o=(y*W+x)*3;color[o]=r;color[o+1]=g;color[o+2]=b;}}
const light=norm([0.4,0.5,-0.9]), baseCol=[150,170,190];
const v0=[0,0,0],v1=[0,0,0],v2=[0,0,0];
for(let t=0;t<nTris;t++){
  const ia=idx[t*3],ib=idx[t*3+1],ic=idx[t*3+2];
  v0[0]=pos[ia*3];v0[1]=pos[ia*3+1];v0[2]=pos[ia*3+2];
  v1[0]=pos[ib*3];v1[1]=pos[ib*3+1];v1[2]=pos[ib*3+2];
  v2[0]=pos[ic*3];v2[1]=pos[ic*3+1];v2[2]=pos[ic*3+2];
  const e1=[v1[0]-v0[0],v1[1]-v0[1],v1[2]-v0[2]],e2=[v2[0]-v0[0],v2[1]-v0[1],v2[2]-v0[2]];
  let n=cross(e1,e2);const nl=Math.hypot(...n)||1;n=[n[0]/nl,n[1]/nl,n[2]/nl];
  const lit=Math.min(1,0.35+0.85*Math.abs(dot(n,light)));
  const cr=Math.min(255,baseCol[0]*lit),cg=Math.min(255,baseCol[1]*lit),cb=Math.min(255,baseCol[2]*lit);
  const p0=project(v0),p1=project(v1),p2=project(v2);
  let minx=Math.floor(Math.min(p0[0],p1[0],p2[0])),maxx=Math.ceil(Math.max(p0[0],p1[0],p2[0]));
  let miny=Math.floor(Math.min(p0[1],p1[1],p2[1])),maxy=Math.ceil(Math.max(p0[1],p1[1],p2[1]));
  if(maxx<0||minx>=W||maxy<0||miny>=H)continue;
  if(minx<0)minx=0;if(miny<0)miny=0;if(maxx>=W)maxx=W-1;if(maxy>=H)maxy=H-1;
  const area=(p1[0]-p0[0])*(p2[1]-p0[1])-(p2[0]-p0[0])*(p1[1]-p0[1]);
  if(Math.abs(area)<1e-9)continue;const invArea=1/area;
  for(let y=miny;y<=maxy;y++)for(let x=minx;x<=maxx;x++){
    const px=x+0.5,py=y+0.5;
    const w0=((p1[0]-px)*(p2[1]-py)-(p2[0]-px)*(p1[1]-py))*invArea;
    const w1=((p2[0]-px)*(p0[1]-py)-(p0[0]-px)*(p2[1]-py))*invArea;
    const w2=1-w0-w1; if(w0<0||w1<0||w2<0)continue;
    const z=w0*p0[2]+w1*p1[2]+w2*p2[2],zi=y*W+x;
    if(z<zbuf[zi]){zbuf[zi]=z;const o=zi*3;color[o]=cr;color[o+1]=cg;color[o+2]=cb;}
  }
}
function png(w,h,rgb){
  const raw=Buffer.alloc((w*3+1)*h);
  for(let y=0;y<h;y++){raw[y*(w*3+1)]=0;Buffer.from(rgb.buffer,rgb.byteOffset+y*w*3,w*3).copy(raw,y*(w*3+1)+1);}
  const comp=zlib.deflateSync(raw,{level:6});
  const ct=(()=>{const t=[];for(let n=0;n<256;n++){let c=n;for(let k=0;k<8;k++)c=c&1?0xedb88320^(c>>>1):c>>>1;t[n]=c>>>0;}return t;})();
  const crc=(b)=>{let c=0xffffffff;for(let i=0;i<b.length;i++)c=ct[(c^b[i])&0xff]^(c>>>8);return (c^0xffffffff)>>>0;};
  const chunk=(ty,d)=>{const len=Buffer.alloc(4);len.writeUInt32BE(d.length,0);const t=Buffer.from(ty);const cd=Buffer.concat([t,d]);const cr=Buffer.alloc(4);cr.writeUInt32BE(crc(cd),0);return Buffer.concat([len,cd,cr]);};
  const sig=Buffer.from([137,80,78,71,13,10,26,10]);
  const ihdr=Buffer.alloc(13);ihdr.writeUInt32BE(w,0);ihdr.writeUInt32BE(h,4);ihdr[8]=8;ihdr[9]=2;
  return Buffer.concat([sig,chunk("IHDR",ihdr),chunk("IDAT",comp),chunk("IEND",Buffer.alloc(0))]);
}
fs.writeFileSync("plant_base.png",png(W,H,Buffer.from(color.buffer,color.byteOffset,color.length)));
// dump camera so the overlay pass projects identically
fs.writeFileSync("camera.json",JSON.stringify({W,H,scale,ctr,right,up,fwd,extents:{lo,hi}}));
console.log("wrote plant_base.png + camera.json  extents lo",lo.map(v=>v.toFixed(1)).join(","),"hi",hi.map(v=>v.toFixed(1)).join(","));
