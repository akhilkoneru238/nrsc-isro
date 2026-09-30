import React, { useCallback, useEffect, useRef, useState } from 'react';
import { UploadCloud, FileImage, ScanSearch, Loader, Layers, RotateCcw, Download, X } from 'lucide-react';
import './industry.css';

const API_BASE = import.meta.env.VITE_API_BASE;

const BG = '#ffffff';
const PANEL = '#ffffff';
const BORDER = '#ffffff';
const ACCENT = '#ffffff';
const TOOLBAR_BG = "#162466";

const Corners = () => (
  <>
    <i className="corner tl" /><i className="corner tr" />
    <i className="corner bl" /><i className="corner br" />
  </>
);

const imgBox = {
  position: 'relative',
  height: 640,
  border: `1px solid ${BORDER}`,
  background: '#ffffff',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  overflow: 'hidden',
};

const panel = {
  position: 'relative',
  padding: 14,
  background: '#ffffff',
  color: '#111827',
};

const panelLabel = {
  fontFamily: 'Nunito, system-ui, sans-serif',
  fontSize: 14,
  letterSpacing: '1px',
  color: '#111827',
};

export default function App() {
  console.log("APP RENDERED");
  const [threshold] = useState(40);
  const [imageSize, setImageSize] = useState(1024);
  const [model, setModel] = useState('yolo');
  const [drag, setDrag] = useState(false);
  const [fileName, setFileName] = useState('');
  const [fileMeta, setFileMeta] = useState('');
  const [hasImage, setHasImage] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [done, setDone] = useState(false);
  const [fullSrc, setFullSrc] = useState('');
  const [lightboxOpen, setLightboxOpen] = useState(false);
const [craterCount, setCraterCount] = useState(0);
const [detections, setDetections] = useState([]);
  const fileRef = useRef(null);
  const inCanvasRef = useRef(null);
  const outCanvasRef = useRef(null);
  const fullSrcRef = useRef('');

  useEffect(() => () => {
    if (fullSrcRef.current) URL.revokeObjectURL(fullSrcRef.current);
  }, []);

  useEffect(() => {
    if (!lightboxOpen) return;
    const onKey = (e) => { if (e.key === 'Escape') setLightboxOpen(false); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [lightboxOpen]);

  const loadFile = useCallback((f) => {
    if (fullSrcRef.current) URL.revokeObjectURL(fullSrcRef.current);
    const url = URL.createObjectURL(f);
    fullSrcRef.current = url;
    setFullSrc(url);
    const img = new Image();
    img.onload = () => {
      const c = inCanvasRef.current;
      c.width = imageSize;
      c.height = imageSize;
      c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
      setHasImage(true);
      setDone(false);
      setFileName(f.name);
      setFileMeta(`${(f.size / 1024).toFixed(0)} KB · ${img.width}×${img.height}`);
    };
    img.src = url;
  }, [imageSize]);

  const openPicker = () => fileRef.current && fileRef.current.click();
  const onInputBoxClick = () => { if (hasImage) setLightboxOpen(true); else openPicker(); };
  const onFile = (e) => { const f = e.target.files?.[0]; if (f) loadFile(f); };
  const onDragOver = (e) => { e.preventDefault(); setDrag(true); };
  const onDragLeave = (e) => { e.preventDefault(); setDrag(false); };
  const onDrop = (e) => {
    e.preventDefault(); setDrag(false);
    const f = e.dataTransfer.files?.[0];
    if (f && f.type.startsWith('image')) loadFile(f);
  };

  const detectViaApi = async (chosen) => {
    const src = inCanvasRef.current;
    const blob = await new Promise((resolve) => src.toBlob(resolve, 'image/png'));
    const fd = new FormData();
    fd.append('image', blob, fileName || 'input.png');
    fd.append('model', chosen === 'rcnn' ? 'faster-rcnn' : 'yolo');
    fd.append('conf', '0.4');
    fd.append('threshold', '0.4');
    fd.append('iou', '0.6');
    fd.append('imgsz', imageSize);
    fd.append('max_det', '3000');
    fd.append('render', 'false');
    fd.append('labels', 'true');

    const res = await fetch(`${API_BASE}/api/detect`, {
      method: "POST",
      body: fd
    });

    if (!res.ok) {
      throw new Error(`API ${res.status}: ${await res.text()}`);
    }

    const data = await res.json();
    setCraterCount(data.count);
    setDetections(data.detections || []);
    const out = outCanvasRef.current;
    out.width = src.width;
    out.height = src.height;

    const ctx = out.getContext("2d");
    ctx.drawImage(src, 0, 0);
    const scale = src.width / data.image_width;
    ctx.strokeStyle = "#00FF00";
    ctx.fillStyle = ctx.strokeStyle;
    ctx.lineWidth = Math.max(1, src.width / 300);
    ctx.font = `${Math.max(9, src.width / 52)}px monospace`;

    (data.detections || []).forEach((d) => {
      ctx.strokeRect(d.x * scale, d.y * scale, d.w * scale, d.h * scale);
    });

    setProcessing(false);
    setDone(true);
  };

  const simulate = (chosen) => {
    const src = inCanvasRef.current;
    const out = outCanvasRef.current;
    const w = src.width;
    const h = src.height;
    out.width = w;
    out.height = h;
    const sctx = src.getContext('2d');
    const octx = out.getContext('2d');
    const data = sctx.getImageData(0, 0, w, h);
    const px = data.data;
    const fine = chosen === 'rcnn';

    for (let i = 0; i < px.length; i += 4) {
      let g = px[i] * 0.299 + px[i + 1] * 0.587 + px[i + 2] * 0.114;
      if (fine) g = Math.min(255, Math.max(0, (g - 128) * 1.5 + 128));
      px[i] = px[i + 1] = px[i + 2] = g * 0.94;
    }
    octx.putImageData(data, 0, 0);

    const conf = threshold / 100;
    const count = Math.max(4, Math.round((fine ? 12 : 6) * (1.2 - conf * 0.3)));
    let seed = 42 + (fine ? 7 : 0);
    const rnd = () => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return seed / 0x7fffffff;
    };
    octx.strokeStyle = '#00FF00';
    octx.fillStyle = octx.strokeStyle;
    octx.lineWidth = Math.max(1, w / 300);
    octx.font = `${Math.max(9, w / 52)}px monospace`;

    for (let k = 0; k < count; k++) {
      const r = w * ((fine ? 0.015 : 0.03) + rnd() * (fine ? 0.06 : 0.09));
      const cx = r + rnd() * (w - 2 * r);
      const cy = r + rnd() * (h - 2 * r);
      octx.beginPath();
      octx.arc(cx, cy, r, 0, Math.PI * 2);
      octx.stroke();
      octx.beginPath();
      octx.moveTo(cx - 4, cy);
      octx.lineTo(cx + 4, cy);
      octx.moveTo(cx, cy - 4);
      octx.lineTo(cx, cy + 4);
      octx.stroke();
      octx.fillText(`#${k + 1}`, cx - r, cy - r - 3);
    }
    setProcessing(false);
    setDone(true);
  };

  const run = (chosen) => {
    if (!hasImage || processing) return;
    setModel(chosen);
    setProcessing(true);
    setDone(false);
    setTimeout(() => {
      detectViaApi(chosen).catch((err) => {
        console.warn('Backend unavailable, using local simulation:', err.message);
        simulate(chosen);
      });
    }, 300);
  };

  const reset = () => {
    [inCanvasRef, outCanvasRef].forEach((r) => {
      const c = r.current;
      if (c) {
        c.width = 0;
        c.height = 0;
      }
    });
    setHasImage(false);
    setDone(false);
    setProcessing(false);
    setFileName('');
    setFileMeta('');
    setCraterCount(0);
    setDetections([]);
if (fullSrcRef.current) URL.revokeObjectURL(fullSrcRef.current);
    fullSrcRef.current = '';
    setFullSrc('');
    setLightboxOpen(false);
  };

  const download = () => {
    const c = outCanvasRef.current;
    if (!c) return;
    const a = document.createElement('a');
    a.download = 'crater-detection-output.png';
    a.href = c.toDataURL('image/png');
    a.click();
  };

const downloadCSV = async () => {
  try {
    const src = inCanvasRef.current;
    const blob = await new Promise((resolve) =>
      src.toBlob(resolve, "image/png")
    );
    const fd = new FormData();
    fd.append("image", blob, fileName || "input.png");
    fd.append("model", model === "rcnn" ? "faster-rcnn" : "yolo");
    fd.append("conf", "0.4");
    fd.append("iou", "0.6");
    fd.append("imgsz", imageSize);
    fd.append("max_det", "3000");

    const response = await fetch(`${API_BASE}/api/detect/csv`, {
      method: "POST",
      body: fd,
    });
    if (!response.ok) {
      throw new Error(await response.text());
    }
    const csvBlob = await response.blob();
    const url = URL.createObjectURL(csvBlob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${fileName.split(".")[0]}_detections.csv`;
    a.click();
    URL.revokeObjectURL(url);
  } catch (err) {
    console.error(err);
    alert("Failed to download CSV.");
  }
};

  return (
    <div style={{
      minHeight: '100vh', background: BG, color: '#111827',
      fontFamily: 'Nunito, system-ui, sans-serif', display: 'flex', flexDirection: 'column',
    }}>
      <header className="nav" style={{
        position: 'relative', zIndex: 2, width: '100%', height: 100,
        borderBottom: `1px solid ${BORDER}`, background: PANEL,
        display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden',
      }}>
        <img
          src="/title.png"
          alt="Lunar Crater Detection Model"
          style={{ width: '100%', height: '100%', objectFit: 'cover' }}
        />
      </header>
      <main style={{
        position: 'relative', flex: 1, display: 'flex', flexDirection: 'column',
        gap: 22, padding: 28, width: '100%', boxSizing: 'border-box', background: BG,
      }}>
        <div style={{
          position: 'relative', zIndex: 1, display: 'flex', flexDirection: 'column',
          gap: 22, maxWidth: 1800, width: '100%', margin: '0 auto',
        }}>
          <div
  style={{
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",

    background: TOOLBAR_BG,
    padding: "16px 24px",
    borderRadius: "8px",

    marginBottom: 20,
  }}
>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <span style={{
                color: '#ffffff', fontSize: 13, fontWeight: 700,
                textTransform: 'uppercase', letterSpacing: '1.2px', fontFamily: 'Merriweather, serif',
              }}>
                Image Size
              </span>
              <input
                type="number"
                value={imageSize}
                onChange={(e) => setImageSize(Number(e.target.value))}
                style={{ width: 90, height: 36, border: `1px solid ${BORDER}`, background: '#ffffff', color: '#111827', padding: '0 8px' }}
              />
            </div>
            {/* Text Plane */}
<h2
  style={{
    color: "#fff",
    margin: 0,
    fontFamily: "Merriweather, serif",
    fontSize: 24,
    fontWeight: 600,

  }}
>
  Lunar Crater Detection Model
</h2>
            <div style={{ display: 'flex', gap: 10 }}>
              <button className="btn" onClick={openPicker} style={{ height: 40, padding: '0 18px' }}>
                Upload ▾
              </button>
<select
  value={model}
  onChange={(e) => setModel(e.target.value)}
  style={{
    height: 40,
    padding: '0 18px',
    border: '1px solid rgba(255,255,255,0.35)',
    background: '#162466',
    color: '#ffffff',
    borderRadius: 5,
    fontWeight: 600,
  }}
>
                <option value="yolo">YOLO</option>
                <option value="rcnn">Faster-RCNN</option>
              </select>
              <button
                className="btn"
                onClick={() => run(model)}
                disabled={!hasImage || processing}
                style={{ height: 40, padding: '0 18px' }}
              >
                Detect
              </button>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 40 }}>
            <div className="blueprint" style={panel}>
              <Corners />
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
                <span style={panelLabel}>Input</span>
              </div>
              <div
                onDragOver={onDragOver}
                onDragLeave={onDragLeave}
                onDrop={onDrop}
                onClick={onInputBoxClick}
                title={hasImage ? 'Click to view full size' : 'Click to browse'}
                style={{
                  ...imgBox,
                  cursor: 'pointer',
                  ...(drag ? { outline: `2px dashed ${ACCENT}`, outlineOffset: -6 } : {}),
                }}
              >
                <canvas ref={inCanvasRef} style={{ maxWidth: '100%', maxHeight: '100%', display: 'block' }} />
                {!hasImage && (
                  <div style={{
                    position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column',
                    alignItems: 'center', justifyContent: 'center', gap: 10, color: '#6e7681',
                  }}>
                    <UploadCloud size={44} strokeWidth={1.4} color={ACCENT} />
                    <div style={{ fontSize: 12 }}>
                      Click to browse or drag &amp; drop&nbsp; .PNG&nbsp; .TIFF&nbsp; .JPG
                    </div>
                  </div>
                )}
                <input ref={fileRef} type="file" accept="image/*" onChange={onFile} style={{ display: 'none' }} />
              </div>
              {fileName && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, paddingTop: 10, fontSize: 12 }}>
                  <FileImage size={15} strokeWidth={1.5} color={ACCENT} />
                  <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{fileName}</span>
                  <span style={{ color: '#6e7681' }}>{fileMeta}</span>
                </div>
              )}
            </div>

            <div className="blueprint" style={panel}>
              <Corners />
<div
  style={{
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 12,
  }}
>
  <div style={{ display: "flex", alignItems: "center", gap: 15 }}>
    <span style={panelLabel}>Output</span>

    {done && (
      <span
        style={{
          fontSize: 15,
          fontWeight: 500,
          color: "#ba2525",
        }}
      >
        Craters Detected: {craterCount}
      </span>
    )}
  </div>

  {done && (
    <div style={{ display: "flex", gap: 10 }}>
      <button
        className="btn btn-secondary"
        onClick={downloadCSV}
        style={{
          display: "flex",
          alignItems: "center",
          gap: 6,
          padding: "5px 11px",
          fontSize: 11.5,
        }}
      >
        <Download size={13} />
        CSV
      </button>

      <button
        className="btn btn-secondary"
        onClick={download}
        style={{
          display: "flex",
          alignItems: "center",
          gap: 6,
          padding: "5px 11px",
          fontSize: 11.5,
        }}
      >
        <Download size={13} />
        Image
      </button>
    </div>
  )}
</div>
              <div
                style={{ ...imgBox, cursor: done ? 'pointer' : 'default' }}
                onClick={() => {
                  if (done) {
                    setFullSrc(outCanvasRef.current.toDataURL('image/png'));
                    setLightboxOpen(true);
                  }
                }}
                title={done ? "Click to view full size" : ""}
              >
                <canvas ref={outCanvasRef} style={{ maxWidth: '100%', maxHeight: '100%', display: 'block' }} />
                {!processing && !done && (
                  <div style={{
                    position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column',
                    alignItems: 'center', justifyContent: 'center', gap: 8, color: '#8b949e',
                  }}>
                    <ScanSearch size={38} strokeWidth={1.4} />
                    <span style={{ fontSize: 12 }}>Awaiting processing</span>
                  </div>
                )}
                {processing && (
                  <div style={{
                    position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column',
                    alignItems: 'center', justifyContent: 'center', gap: 10,
                    background: 'color-mix(in srgb, #05070a 78%, transparent)',
                  }}>
                    <Loader size={32} strokeWidth={1.6} color={ACCENT} className="spin" />
                    <span style={{ fontSize: 12, letterSpacing: '.5px', color: ACCENT }}>Analyzing…</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </main>

      <footer style={{
        padding: '12px 28px', borderTop: `1px solid ${BORDER}`, fontSize: 10.5,
        letterSpacing: '.4px', color: '#8b949e', textAlign: 'center',
      }}>
        &nbsp;
      </footer>

      {lightboxOpen && fullSrc && (
        <div
          onClick={() => setLightboxOpen(false)}
          style={{
            position: 'fixed', inset: 0, zIndex: 50,
            background: 'rgba(5,7,10,.92)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            cursor: 'zoom-out', padding: 40, boxSizing: 'border-box',
          }}
        >
          <button
            onClick={(e) => {
              e.stopPropagation();
              setLightboxOpen(false);
            }}
            title="Close"
            style={{
              position: 'absolute', top: 20, right: 24, background: 'transparent',
              border: `1px solid ${BORDER}`, color: '#e6edf3', width: 36, height: 36,
              display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer',
            }}
          >
            <X size={18} strokeWidth={1.8} />
          </button>
          <img
            src={fullSrc}
            alt={fileName || 'Original upload'}
            onClick={(e) => e.stopPropagation()}
            style={{
              maxWidth: '100%', maxHeight: '100%', objectFit: 'contain',
              border: `1px solid ${BORDER}`, cursor: 'default',
            }}
          />
        </div>
      )}
    </div>
  );
}