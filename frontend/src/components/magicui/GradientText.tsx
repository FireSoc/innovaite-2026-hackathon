"use client";

import { cn } from "@/lib/utils";

interface GradientTextProps {
  children: React.ReactNode;
  className?: string;
  colors?: string[];
  animationSpeed?: number;
}

export function GradientText({
  children,
  className,
  colors = ["#F59E0B", "#D97706", "#10B981", "#F59E0B"],
  animationSpeed = 8,
}: GradientTextProps) {
  const gradient = `linear-gradient(to right, ${colors.join(", ")})`;

  return (
    <span
      className={cn(
        "bg-clip-text text-transparent animate-gradient",
        className
      )}
      style={{
        backgroundImage: gradient,
        backgroundSize: `${colors.length * 100}% 100%`,
        animationDuration: `${animationSpeed}s`,
      }}
    >
      {children}
    </span>
  );
}
