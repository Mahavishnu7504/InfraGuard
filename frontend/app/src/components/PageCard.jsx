import "./PageCard.css";

export default function PageCard({
  title,
  subtitle,
  children,
  className = "",
}) {
  return (
    <div className={`minimal-card ${className}`}>
      {(title || subtitle) && (
        <div className="minimal-card-header">
          {title && <h3>{title}</h3>}
          {subtitle && <p>{subtitle}</p>}
        </div>
      )}

      <div className="minimal-card-body">
        {children}
      </div>
    </div>
  );
}