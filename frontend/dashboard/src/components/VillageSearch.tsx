import { useEffect, useMemo, useRef, useState } from "react";

export interface SearchEntry {
  id: number;
  name: string;
  subDistrict: string;
  lat: number;
  lon: number;
}

interface Props {
  entries: SearchEntry[];
  onSelect: (entry: SearchEntry) => void;
}

/** Find a village by name or taluka instead of only clicking the map - all 2,003 villages are already
 * loaded client-side with the map data, so this is a pure client-side filter, no extra API call. */
export default function VillageSearch({ entries, onSelect }: Props) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onClickOutside = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, [open]);

  const matches = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (q.length < 2) return [];
    return entries.filter((e) => e.name.toLowerCase().includes(q) || e.subDistrict.toLowerCase().includes(q)).slice(0, 8);
  }, [entries, query]);

  return (
    <div className="village-search" ref={containerRef}>
      <div className="village-search-input-wrap">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
          <circle cx="11" cy="11" r="7" />
          <path d="M21 21l-4.3-4.3" />
        </svg>
        <input
          type="text"
          className="mono"
          placeholder="Search village or taluka..."
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setOpen(true);
          }}
          onFocus={() => setOpen(true)}
        />
      </div>

      {open && query.trim().length >= 2 && (
        <div className="village-search-popover">
          {matches.length === 0 ? (
            <p className="village-search-empty">No village matches &quot;{query}&quot;</p>
          ) : (
            matches.map((m) => (
              <button
                key={m.id}
                type="button"
                className="village-search-result"
                onClick={() => {
                  onSelect(m);
                  setQuery(m.name);
                  setOpen(false);
                }}
              >
                <span className="village-search-name">{m.name}</span>
                <span className="village-search-sub mono">{m.subDistrict}</span>
              </button>
            ))
          )}
        </div>
      )}
    </div>
  );
}
