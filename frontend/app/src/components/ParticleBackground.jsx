import { memo, useEffect, useRef } from "react";
import Particles, { initParticlesEngine } from "@tsparticles/react";
import { loadSlim } from "@tsparticles/slim";

/*
  ParticleBackground renders BELOW everything (z-index: 0).
  index.css sets html/body background to a deep-space gradient.
  All cards use glass (rgba + backdrop-filter) so particles show through.
*/

const PARTICLE_OPTIONS = {
  fullScreen: { enable: false },
  background: { color: "transparent" },
  fpsLimit: 60,

  particles: {
    number: {
      value: 55,
      density: { enable: true, area: 1000 },
    },
    color: {
      value: ["#00c8ff", "#3b82f6", "#60a5fa"],
    },
    links: {
      enable: true,
      color: "#00c8ff",
      distance: 140,
      opacity: 0.1,
      width: 1,
    },
    move: {
      enable: true,
      speed: 0.8,
      outModes: { default: "out" },
    },
    opacity: { value: 0.22 },
    size: { value: { min: 1, max: 2.5 } },
  },

  interactivity: {
    events: {
      onHover: { enable: true, mode: "grab" },
      resize: { enable: true },
    },
    modes: {
      grab: {
        distance: 160,
        links: { opacity: 0.3 },
      },
    },
  },

  detectRetina: true,
};

function ParticleBackground() {
  const ready = useRef(false);

  useEffect(() => {
    if (ready.current) return;
    initParticlesEngine(async (engine) => {
      await loadSlim(engine);
    });
    ready.current = true;
  }, []);

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 0,
        pointerEvents: "none",
      }}
    >
      <Particles
        id="tsparticles"
        options={PARTICLE_OPTIONS}
      />
    </div>
  );
}

export default memo(ParticleBackground);