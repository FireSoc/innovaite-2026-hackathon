"use client";

import { cn } from "@/lib/utils";

interface RippleEffectProps {
  className?: string;
  mainCircleSize?: number;
  mainCircleOpacity?: number;
  numCircles?: number;
  color?: string;
}

export function RippleEffect({
  className,
  mainCircleSize = 210,
  mainCircleOpacity = 0.24,
  numCircles = 8,
  color = "#F59E0B",
}: RippleEffectProps) {
  return (
    <div
      className={cn(
        "pointer-events-none absolute inset-0 select-none [mask-image:linear-gradient(to_bottom,white,transparent)]",
        className
      )}
    >
      {Array.from({ length: numCircles }, (_, i) => {
        const size = mainCircleSize + i * 70;
        const opacity = mainCircleOpacity - i * 0.03;
        const animationDelay = `${i * 0.06}s`;
        return (
          <div
            key={i}
            className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 animate-[ripple_2.5s_ease-in-out_infinite] rounded-full border"
            style={{
              width: `${size}px`,
              height: `${size}px`,
              opacity: Math.max(0, opacity),
              borderColor: color,
              animationDelay,
            }}
          />
        );
      })}
    </div>
  );
}
