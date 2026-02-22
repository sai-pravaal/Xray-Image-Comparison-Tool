# database.py
# SQLite data layer. Ingest folder tree and lookup references (strict side; gender/age fallback allowed).
import os, re, sqlite3
from pathlib import Path
from datetime import datetime

DB_NAME = "xray.db"

BODY_PARTS_CANON = {
    'chest':'chest','femur':'femur','foot':'foot','forearm':'forearm','hand':'hand',
    'humerus':'humerus','shoulder':'shoulder','tibia and fibula':'tibia_fibula','tibia_fibula':'tibia_fibula','wrist':'wrist'
}
SIDES   = {'left','right','midline','unknown'}
GENDERS = {'male','female','unknown'}
IMG_EXTS = {'.dcm','.dicom','.png','.jpg','.jpeg','.tif','.tiff','.bmp'}

AGE_BANDS = [
    (1,'0-10', 0,10),
    (2,'10-20',10,20),
    (3,'20-40',20,40),
    (4,'40-70',40,70),
]

# ---------- connection ----------

def get_conn():
    return sqlite3.connect(DB_NAME)

# ---------- schema ----------

def init_db():
    with get_conn() as conn:
        c=conn.cursor(); c.execute("PRAGMA foreign_keys=ON")
        c.execute("CREATE TABLE IF NOT EXISTS anatomy(body_part TEXT PRIMARY KEY)")
        c.execute("""
            CREATE TABLE IF NOT EXISTS age_band(
              id INTEGER PRIMARY KEY, label TEXT UNIQUE, min_age INTEGER NOT NULL, max_age INTEGER NOT NULL)
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS image_store(
              id INTEGER PRIMARY KEY,
              path TEXT NOT NULL UNIQUE,
              body_part TEXT NOT NULL,
              side TEXT NOT NULL CHECK(side IN ('left','right','midline','unknown')),
              gender TEXT NOT NULL CHECK(gender IN ('male','female','unknown')),
              age INTEGER NULL,
              added_at TEXT NOT NULL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS reference_map(
              image_id INTEGER PRIMARY KEY,
              is_perfect INTEGER NOT NULL DEFAULT 0,
              notes TEXT,
              FOREIGN KEY(image_id) REFERENCES image_store(id) ON DELETE CASCADE
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_lookup ON image_store(body_part, side, gender, age)")
        # seed anatomy & age bands
        c.execute("SELECT COUNT(*) FROM anatomy");
        if c.fetchone()[0]==0:
            c.executemany("INSERT INTO anatomy(body_part) VALUES (?)", [(v,) for v in sorted(set(BODY_PARTS_CANON.values()))])
        c.execute("DELETE FROM age_band")
        c.executemany("INSERT INTO age_band(id,label,min_age,max_age) VALUES (?,?,?,?)", AGE_BANDS)
        conn.commit()

# ---------- helpers ----------

def parse_age_band_from_folder(name: str):
    name=name.strip()
    for _i, label, lo, hi in AGE_BANDS:
        if name==label:
            return (lo,hi)
    m=re.match(r"^(\d+)[-_](\d+)$", name)
    if m:
        lo=int(m.group(1)); hi=int(m.group(2))
        if lo < 10: return (0,10)
        if lo < 20: return (10,20)
        if lo < 40: return (20,40)
        return (40,70)
    return None

# ---------- ingest ----------

def ingest_tree(root='data'):
    root_p=Path(root)
    if not root_p.exists():
        return {'inserted':0,'skipped':0}
    ins=0; sk=0
    with get_conn() as conn:
        c=conn.cursor()
        for body_dir in sorted(root_p.iterdir()):
            if not body_dir.is_dir(): continue
            display=body_dir.name
            body=BODY_PARTS_CANON.get(display, BODY_PARTS_CANON.get(display.lower()))
            if not body: continue
            # expect body/side/gender/age-band/files
            for side_dir in sorted(body_dir.iterdir()):
                if not side_dir.is_dir(): continue
                side=side_dir.name.lower()
                if side not in SIDES:
                    # tree without explicit side level
                    gender_levels=[side_dir]; side='unknown'
                else:
                    gender_levels=[p for p in sorted(side_dir.iterdir()) if p.is_dir()]
                for gender_dir in gender_levels:
                    gender=gender_dir.name.lower()
                    if gender not in GENDERS:
                        age_levels=[gender_dir]; gender='unknown'
                    else:
                        age_levels=[p for p in sorted(gender_dir.iterdir()) if p.is_dir()]
                    for age_dir in age_levels:
                        rng=parse_age_band_from_folder(age_dir.name)
                        if rng is None:
                            lo=hi=None
                        else:
                            lo,hi=rng
                        for f in age_dir.rglob('*'):
                            if f.is_file() and f.suffix.lower() in IMG_EXTS:
                                age_val=None
                                if lo is not None and hi is not None:
                                    age_val=(lo+hi)//2
                                try:
                                    c.execute(
                                        "INSERT INTO image_store(path,body_part,side,gender,age,added_at) VALUES (?,?,?,?,?,?)",
                                        (str(f), body, side, gender, age_val, datetime.utcnow().isoformat())
                                    ); ins+=1
                                except sqlite3.IntegrityError:
                                    sk+=1
        conn.commit()
    return {'inserted':ins,'skipped':sk}

# ---------- mark perfect (optional) ----------

def set_perfect(path: str, is_perfect=True, notes: str | None = None):
    with get_conn() as conn:
        c=conn.cursor(); c.execute("SELECT id FROM image_store WHERE path=?", (path,)); row=c.fetchone()
        if not row: raise ValueError(f"Not found in DB: {path}")
        image_id=row[0]
        c.execute(
            "INSERT INTO reference_map(image_id,is_perfect,notes) VALUES (?,?,?)\n             ON CONFLICT(image_id) DO UPDATE SET is_perfect=excluded.is_perfect, notes=excluded.notes",
            (image_id, 1 if is_perfect else 0, notes)
        ); conn.commit()

# ---------- lookup with strict-side & gender/age fallback ----------

def _band_for_age(age: int | None):
    if age is None: return None
    for _id,_lab,lo,hi in AGE_BANDS:
        if lo <= age <= hi: return (lo,hi)
    return None


def find_reference(body_part: str, side: str, gender: str, age: int | None):
    """Return (id, path, age) or None.
    Order (always same side):
      1) same gender + same band (perfect)
      2) any gender  + same band (perfect)
      3) same gender + nearest band (perfect)
      4) any gender  + nearest band (perfect)
      5-8) repeat 1-4 but allow any image (not only perfect)
    """
    band=_band_for_age(age) if age is not None else None
    with get_conn() as conn:
        c=conn.cursor()
        def pick(lo,hi,genders,require_perfect=True):
            if require_perfect:
                q=(
                    "SELECT s.id,s.path,s.age FROM image_store s "
                    "JOIN reference_map r ON r.image_id = s.id AND r.is_perfect = 1 "
                    "WHERE s.body_part=? AND s.side=? AND s.gender IN (%s)" % (','.join('?'*len(genders)))
                )
            else:
                q=(
                    "SELECT s.id,s.path,s.age FROM image_store s "
                    "WHERE s.body_part=? AND s.side=? AND s.gender IN (%s)" % (','.join('?'*len(genders)))
                )
            params=[body_part, side] + list(genders)
            if lo is not None and hi is not None:
                q += " AND (s.age BETWEEN ? AND ? OR s.age IS NULL)"; params += [lo,hi]
            q += " ORDER BY CASE WHEN s.age IS NULL THEN 1 ELSE 0 END, ABS(COALESCE(s.age,0)-?) ASC LIMIT 1"; params += [age or 0]
            c.execute(q, params); return c.fetchone()
        def neighbors(cur):
            bands=[(lo,hi) for _,_,lo,hi in AGE_BANDS]
            if not cur: return bands
            lo,hi=cur; ctr=(lo+hi)/2
            return [b for b in sorted(bands, key=lambda b: abs(((b[0]+b[1])/2)-ctr)) if b!=cur]
        for require_perfect in (True, False):
            if band:
                lo,hi=band
                row=pick(lo,hi,(gender,),require_perfect);   
                if row: return row
                row=pick(lo,hi,tuple(GENDERS),require_perfect);
                if row: return row
                for nlo,nhi in neighbors(band):
                    row=pick(nlo,nhi,(gender,),require_perfect); 
                    if row: return row
                    row=pick(nlo,nhi,tuple(GENDERS),require_perfect); 
                    if row: return row
            else:
                row=pick(None,None,(gender,),require_perfect); 
                if row: return row
                row=pick(None,None,tuple(GENDERS),require_perfect); 
                if row: return row
        return None
