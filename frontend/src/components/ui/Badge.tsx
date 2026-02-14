"use client";

import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";
import { type HTMLAttributes } from "react";

const badgeVariants = cva(
  "inline-flex items-center gap-1.5 font-medium rounded-full whitespace-nowrap transition-colors",
  {
    variants: {
      variant: {
        default: "bg-secondary text-secondary-foreground",
        success: "bg-success-light text-success border border-success/20",
        danger: "bg-danger-light text-danger border border-danger/20",
        warning: "bg-amber-light/40 text-amber-dark border border-amber/20",
        info: "bg-info-light text-info border border-info/20",
        outline: "border border-border text-foreground",
      },
      size: {
        sm: "text-xs px-2 py-0.5",
        md: "text-sm px-2.5 py-0.5",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "sm",
    },
  }
);

interface BadgeProps
  extends HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {
  dot?: boolean;
}

export function Badge({
  variant = "default",
  size = "sm",
  dot,
  className,
  children,
  ...props
}: BadgeProps) {
  return (
    <span
      className={cn(badgeVariants({ variant, size }), className)}
      {...props}
    >
      {dot && (
        <span
          className={cn(
            "h-1.5 w-1.5 rounded-full",
            variant === "success" && "bg-success",
            variant === "danger" && "bg-danger",
            variant === "warning" && "bg-amber",
            variant === "info" && "bg-info",
            variant === "default" && "bg-muted-foreground",
            variant === "outline" && "bg-muted-foreground"
          )}
        />
      )}
      {children}
    </span>
  );
}

export { badgeVariants };
