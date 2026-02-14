"use client";

import { cn } from "@/lib/utils";
import { type ButtonHTMLAttributes, forwardRef } from "react";

interface ShimmerButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  shimmerColor?: string;
  shimmerSize?: string;
  borderRadius?: string;
  shimmerDuration?: string;
  background?: string;
}

const ShimmerButton = forwardRef<HTMLButtonElement, ShimmerButtonProps>(
  (
    {
      shimmerColor = "#ffffff",
      shimmerSize = "0.1em",
      borderRadius = "1rem",
      shimmerDuration = "2s",
      background = "linear-gradient(135deg, #F59E0B 0%, #D97706 100%)",
      className,
      children,
      ...props
    },
    ref
  ) => {
    return (
      <button
        ref={ref}
        className={cn(
          "group relative inline-flex h-14 cursor-pointer items-center justify-center gap-3 overflow-hidden whitespace-nowrap px-8 text-lg font-semibold text-white shadow-lg transition-all duration-300 hover:shadow-xl active:scale-[0.98] disabled:pointer-events-none disabled:opacity-50",
          className
        )}
        style={{
          borderRadius,
          background,
        }}
        {...props}
      >
        {/* Shimmer effect */}
        <div
          className="absolute inset-0 overflow-hidden"
          style={{ borderRadius }}
        >
          <div
            className="absolute inset-[-100%] animate-[shimmer_2s_linear_infinite]"
            style={{
              background: `linear-gradient(90deg, transparent 20%, ${shimmerColor}20 50%, transparent 80%)`,
              animationDuration: shimmerDuration,
            }}
          />
        </div>

        {/* Glow effect on hover */}
        <div
          className="absolute inset-0 opacity-0 transition-opacity duration-300 group-hover:opacity-100"
          style={{
            borderRadius,
            background: `radial-gradient(ellipse at center, ${shimmerColor}15 0%, transparent 70%)`,
          }}
        />

        {/* Content */}
        <span className="relative z-10 flex items-center gap-3">{children}</span>
      </button>
    );
  }
);

ShimmerButton.displayName = "ShimmerButton";
export { ShimmerButton };
