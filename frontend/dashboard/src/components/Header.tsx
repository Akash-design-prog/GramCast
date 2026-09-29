import DatePicker from "./DatePicker";
import VillageSearch, { type SearchEntry } from "./VillageSearch";

interface Props {
  selectedDate: string;
  onDateChange: (date: string) => void;
  availableDates: string[];
  searchEntries: SearchEntry[];
  onSearchSelect: (entry: SearchEntry) => void;
  theme: "light" | "dark";
  onToggleTheme: () => void;
}

export default function Header({
  selectedDate,
  onDateChange,
  availableDates,
  searchEntries,
  onSearchSelect,
  theme,
  onToggleTheme,
}: Props) {
  return (
    <header className="header">
      <div className="brand">
        <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
          <path d="M20 2 L36 11 V29 L20 38 L4 29 V11 Z" fill="var(--accent)" />
          <path
            d="M20 10 C24 17 27 21 27 25 C27 29.4 23.9 32 20 32 C16.1 32 13 29.4 13 25 C13 21 16 17 20 10 Z"
            fill="var(--surface)"
          />
        </svg>
        <div className="brand-text">
          <h1>GramCast</h1>
          <p>Panchayat rainfall intelligence &mdash; Pune district</p>
        </div>
      </div>
      <div className="header-actions">
        <VillageSearch entries={searchEntries} onSelect={onSearchSelect} />
        <DatePicker value={selectedDate} onChange={onDateChange} availableDates={availableDates} />
        <a
          className="icon-link"
          href="https://github.com/Akash-design-prog/GramCast"
          target="_blank"
          rel="noopener noreferrer"
          aria-label="GramCast on GitHub"
          title="View on GitHub"
        >
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12" />
          </svg>
        </a>
        <a className="docs-link" href="https://akash-design-prog.github.io/gramcast-docs/" target="_blank" rel="noopener noreferrer">
          Docs
        </a>
        <button className="theme-btn" onClick={onToggleTheme} aria-label="Toggle theme" title="Toggle theme">
          {theme === "dark" ? (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <circle cx="12" cy="12" r="4" />
              <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
            </svg>
          )}
        </button>
      </div>
    </header>
  );
}
