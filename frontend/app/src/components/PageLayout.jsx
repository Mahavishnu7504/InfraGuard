import { memo } from "react";
import Sidebar from "./Sidebar";
import "./PageLayout.css";

function PageLayout({
    badge = "",
    title = "",
    subtitle = "",
    rightContent = null,
    children,
}) {
    return (
        <div className="app-shell">
            <Sidebar />

            <div className="main-layout">
                <main className="page-wrapper" role="main">
                    <div className="page-container">

                        {(badge || title || subtitle || rightContent) && (
                            <section className="minimal-page-header fade-in">
                                <div className="minimal-page-header-left">
                                    {badge && <div className="minimal-badge">{badge}</div>}
                                    {title && <h1 className="minimal-page-title">{title}</h1>}
                                    {subtitle && <p className="minimal-page-subtitle">{subtitle}</p>}
                                </div>

                                {rightContent && (
                                    <div className="minimal-page-right">{rightContent}</div>
                                )}
                            </section>
                        )}

                        <section className="page-body fade-in">
                            {children}
                        </section>

                    </div>
                </main>
            </div>
        </div>
    );
}

export default memo(PageLayout);