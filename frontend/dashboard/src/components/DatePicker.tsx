import { useEffect, useMemo, useRef, useState } from "react";

const WEEKDAY_LABELS = ["S", "M", "T", "W", "T", "F", "S"];
const MONTH_LABELS = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December",
];

function parseUTC(date: string): Date {
  return new Date(date + "T00:00:00Z");
}

function formatLong(date: string): string {
  const d = parseUTC(date);
  return `${d.getUTCDate()} ${MONTH_LABELS[d.getUTCMonth()].slice(0, 3)} ${d.getUTCFullYear()}`;
}

interface Props {
  value: string;
  onChange: (date: string) => void;
  availableDates: string[];
}

/** A real calendar picker constrained to dates the model actually has predictions for - the dataset is
 * monsoon-season-only (Jun-Sep) across 1981-2026 with real gaps, so this doesn't just clamp a min/max
 * range, it disables every individual day not present in `availableDates` (backend's own /dates list). */
export default function DatePicker({ value, onChange, availableDates }: Props) {
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const availableSet = useMemo(() => new Set(availableDates), [availableDates]);
  const minDate = availableDates[0];
  const maxDate = availableDates[availableDates.length - 1];
  const minYear = minDate ? parseUTC(minDate).getUTCFullYear() : new Date().getUTCFullYear();
  const maxYear = maxDate ? parseUTC(maxDate).getUTCFullYear() : new Date().getUTCFullYear();
  const years = useMemo(() => {
    const list: number[] = [];
    for (let y = maxYear; y >= minYear; y--) list.push(y); // newest first - more useful default scan order
    return list;
  }, [minYear, maxYear]);

  const initial = parseUTC(availableSet.has(value) ? value : (maxDate ?? value));
  const [viewYear, setViewYear] = useState(initial.getUTCFullYear());
  const [viewMonth, setViewMonth] = useState(initial.getUTCMonth());

  useEffect(() => {
    if (!open) return;
    const onClickOutside = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, [open]);

  if (availableDates.length === 0) {
    return <span className="date-picker-loading mono">Loading dates...</span>;
  }

  const firstOfMonth = new Date(Date.UTC(viewYear, viewMonth, 1));
  const daysInMonth = new Date(Date.UTC(viewYear, viewMonth + 1, 0)).getUTCDate();
  const startWeekday = firstOfMonth.getUTCDay();

  const canGoPrev = new Date(Date.UTC(viewYear, viewMonth, 1)) > parseUTC(minDate);
  const canGoNext = new Date(Date.UTC(viewYear, viewMonth + 1, 1)) <= parseUTC(maxDate);

  function goMonth(delta: number) {
    let m = viewMonth + delta;
    let y = viewYear;
    if (m < 0) {
      m = 11;
      y -= 1;
    } else if (m > 11) {
      m = 0;
      y += 1;
    }
    setViewMonth(m);
    setViewYear(y);
  }

  const cells: (string | null)[] = Array.from({ length: startWeekday }, () => null);
  for (let day = 1; day <= daysInMonth; day++) {
    const iso = `${viewYear}-${String(viewMonth + 1).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
    cells.push(iso);
  }

  return (
    <div className="date-picker" ref={containerRef}>
      <button type="button" className="date-picker-trigger mono" onClick={() => setOpen((o) => !o)}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
          <rect x="3" y="5" width="18" height="16" rx="2" />
          <path d="M3 10h18M8 3v4M16 3v4" />
        </svg>
        {formatLong(value)}
      </button>

      {open && (
        <div className="date-picker-popover">
          <div className="date-picker-nav">
            <button type="button" disabled={!canGoPrev} onClick={() => goMonth(-1)} aria-label="Previous month">
              &lsaquo;
            </button>
            <span className="date-picker-nav-label">
              {MONTH_LABELS[viewMonth]}{" "}
              <select
                className="date-picker-year-select mono"
                value={viewYear}
                onChange={(e) => setViewYear(Number(e.target.value))}
                aria-label="Jump to year"
              >
                {years.map((y) => (
                  <option key={y} value={y}>
                    {y}
                  </option>
                ))}
              </select>
            </span>
            <button type="button" disabled={!canGoNext} onClick={() => goMonth(1)} aria-label="Next month">
              &rsaquo;
            </button>
          </div>
          <div className="date-picker-weekdays">
            {WEEKDAY_LABELS.map((w, i) => (
              <span key={i}>{w}</span>
            ))}
          </div>
          <div className="date-picker-grid">
            {cells.map((iso, i) => {
              if (iso === null) return <span key={`blank-${i}`} />;
              const isAvailable = availableSet.has(iso);
              const isSelected = iso === value;
              return (
                <button
                  key={iso}
                  type="button"
                  disabled={!isAvailable}
                  className={`date-cell${isSelected ? " selected" : ""}${isAvailable ? "" : " unavailable"}`}
                  onClick={() => {
                    onChange(iso);
                    setOpen(false);
                  }}
                >
                  {Number(iso.slice(8, 10))}
                </button>
              );
            })}
          </div>
          <p className="date-picker-hint">Dataset covers Jun-Sep, {minDate.slice(0, 4)}-{maxDate.slice(0, 4)}</p>
        </div>
      )}
    </div>
  );
}
