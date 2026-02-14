"use client";

import { cn } from "@/lib/utils";

interface BorderBeamProps {
  className?: string;
  size?: number;
  duration?: number;
  delay?: number;
  colorFrom?: string;
  colorTo?: string;
  borderWidth?: number;
}

export function BorderBeam({
  className,
  size = 200,
  duration = 12,
  delay = 0,
  colorFrom = "#F59E0B",
  colorTo = "#10B981",
  borderWidth = 2,
}: BorderBeamProps) {
  return (
    <div
      className={cn(
        "pointer-events-none absolute inset-0 rounded-[inherit]",
        className
      )}
      style={{
        borderWidth,
        borderStyle: "solid",
        borderColor: "transparent",
        WebkitMask:
          "linear-gradient(#fff 0 0) padding-box, linear-gradient(#fff 0 0)",
        WebkitMaskComposite: "xor",
        maskComposite: "exclude",
        backgroundImage: `conic-gradient(from calc(var(--border-beam-start, 0) * 1turn), transparent 0%, ${colorFrom} 10%, ${colorTo} 20%, transparent 30%)`,
        backgroundOrigin: "border-box",
        backgroundClip: "border-box",
        animation: `border-beam ${duration}s linear ${delay}s infinite`,
        ["--border-beam-size" as string]: `${size}px`,
      }}
    />
  );
}
