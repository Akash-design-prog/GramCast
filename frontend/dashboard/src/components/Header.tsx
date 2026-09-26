interface Props {
  dayLabels: string[];
  activeDay: number;
  onDayChange: (index: number) => void;
  theme: "light" | "dark";
  onToggleTheme: () => void;
}

export default function Header({ dayLabels, activeDay, onDayChange, theme, onToggleTheme }: Props) {
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
        <div className="day-pills">
          {dayLabels.map((label, i) => (
            <button key={label} className={`day-pill${i === activeDay ? " active" : ""}`} onClick={() => onDayChange(i)}>
              {label}
            </button>
          ))}
        </div>
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
