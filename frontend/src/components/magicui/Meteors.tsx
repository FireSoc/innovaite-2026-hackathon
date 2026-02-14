"use client";

import { cn } from "@/lib/utils";

interface MeteorsProps {
  number?: number;
  className?: string;
}

export function Meteors({ number = 20, className }: MeteorsProps) {
  const meteors = [...Array(number)].map((_, i) => ({
    id: i,
    left: `${Math.random() * 100}%`,
    animationDelay: `${Math.random() * 3}s`,
    animationDuration: `${Math.random() * 3 + 2}s`,
    width: `${Math.random() * 2 + 1}px`,
  }));

  return (
    <div className={cn("absolute inset-0 overflow-hidden", className)} aria-hidden="true">
      {meteors.map((meteor) => (
        <span
          key={meteor.id}
          className="absolute top-1/2 left-1/2 h-0.5 rotate-[215deg] rounded-full bg-gradient-to-r from-amber/0 via-amber/50 to-amber shadow-[0_0_0_1px_#ffffff10] animate-meteor"
          style={{
            left: meteor.left,
            top: "-5%",
            animationDelay: meteor.animationDelay,
            animationDuration: meteor.animationDuration,
            width: "150px",
          }}
        >
          {/* Tail glow */}
          <span className="absolute top-1/2 -translate-y-1/2 left-0 h-[1px] w-[50px] bg-gradient-to-r from-amber/80 to-transparent" />
        </span>
      ))}
    </div>
  );
}
