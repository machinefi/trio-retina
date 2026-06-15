import { IModelHost, SnapshotDb } from "@itwin/core-backend";
import { DbResult } from "@itwin/core-bentley";
import fs from "fs";

const main = async () => {
  await IModelHost.startup();
  const db = SnapshotDb.openFile("./Baytown.bim");
  const ids = [];
  db.withPreparedStatement("SELECT ECInstanceId FROM bis.GeometricElement3d", (s)=>{
    while (s.step() === DbResult.BE_SQLITE_ROW) ids.push(s.getValue(0).getId());
  });

  // Accumulate into flat arrays. Keep per-triangle so we can shade.
  const positions = []; // x,y,z per vertex
  const tris = [];      // index triples
  let base = 0;
  db.exportGraphics({
    elementIdArray: ids,
    chordTol: 0.02,
    onGraphics: (info) => {
      const m = info.mesh;
      for (let i=0;i<m.points.length;i++) positions.push(m.points[i]);
      for (let i=0;i<m.indices.length;i++) tris.push(m.indices[i] + base);
      base += m.points.length/3;
    },
  });
  db.close();
  await IModelHost.shutdown();

  // write compact binary: header + float32 positions + uint32 indices
  const pos = Float32Array.from(positions);
  const idx = Uint32Array.from(tris);
  const header = Buffer.alloc(8);
  header.writeUInt32LE(pos.length/3, 0);
  header.writeUInt32LE(idx.length/3, 4);
  const out = Buffer.concat([header, Buffer.from(pos.buffer), Buffer.from(idx.buffer)]);
  fs.writeFileSync("scene.bin", out);
  console.log("wrote scene.bin verts:", pos.length/3, "tris:", idx.length/3, "bytes:", out.length);
};
main().catch(e=>{ console.error("ERROR:", e); process.exit(1); });
