"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/cn";

interface ProgressRingProps {
  /** Value from 0-100 */
  value: number;
  /** Size in px */
  size?: number;
  /** Stroke width in px */
  strokeWidth?: number;
  /** Color variant */
  variant?: "amber" | "success" | "danger" | "info";
  /** Label shown inside the ring */
  label?: string;
  /** Sub-label below the main label */
  sublabel?: string;
  className?: string;
}

const colorMap = {
  amber: { stroke: "var(--color-amber)", bg: "var(--color-amber-light)" },
  success: { stroke: "var(--color-success)", bg: "var(--color-success-light)" },
  danger: { stroke: "var(--color-danger)", bg: "var(--color-danger-light)" },
  info: { stroke: "var(--color-info)", bg: "var(--color-info-light)" },
};

export function ProgressRing({
  value,
  size = 200,
  strokeWidth = 12,
  variant = "amber",
  label,
  sublabel,
  className,
}: ProgressRingProps) {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const clampedValue = Math.min(100, Math.max(0, value));
  const offset = circumference - (clampedValue / 100) * circumference;
  const colors = colorMap[variant];

  return (
    <div className={cn("relative inline-flex items-center justify-center", className)}>
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        className="-rotate-90"
      >
        {/* Background circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="var(--color-border-light)"
          strokeWidth={strokeWidth}
        />
        {/* Progress circle */}
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={colors.stroke}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 1.2, ease: [0.16, 1, 0.3, 1] }}
        />
      </svg>
      {/* Center content */}
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        {label && (
          <motion.span
            className="text-3xl font-bold text-text font-mono tracking-tight"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.4, duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
          >
            {label}
          </motion.span>
        )}
        {sublabel && (
          <motion.span
            className="text-sm text-text-muted mt-0.5"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.6, duration: 0.4 }}
          >
            {sublabel}
          </motion.span>
        )}
      </div>
    </div>
  );
}
