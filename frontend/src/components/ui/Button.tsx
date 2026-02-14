"use client";

import { forwardRef, type ButtonHTMLAttributes } from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";
import { Loader2 } from "lucide-react";

const buttonVariants = cva(
  "inline-flex items-center justify-center font-medium cursor-pointer transition-all duration-200 outline-none focus-visible:ring-2 focus-visible:ring-ring/50 focus-visible:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed disabled:pointer-events-none [&_svg]:pointer-events-none [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        primary:
          "bg-primary text-primary-foreground hover:bg-amber-dark shadow-md hover:shadow-lg active:scale-[0.98]",
        secondary:
          "bg-card text-card-foreground border border-border hover:bg-accent hover:border-ring/30 shadow-sm",
        ghost:
          "text-muted-foreground hover:text-foreground hover:bg-accent transition-colors",
        danger:
          "bg-destructive text-white hover:bg-destructive/90 shadow-sm active:scale-[0.98]",
        success:
          "bg-success text-white hover:bg-success/90 shadow-sm active:scale-[0.98]",
        outline:
          "border border-border bg-background shadow-xs hover:bg-accent hover:text-accent-foreground",
      },
      size: {
        sm: "h-8 px-3 text-sm gap-1.5 rounded-lg",
        md: "h-10 px-4 text-sm gap-2 rounded-xl",
        lg: "h-12 px-6 text-base gap-2.5 rounded-xl",
        xl: "h-14 px-8 text-lg gap-3 rounded-2xl",
      },
    },
    defaultVariants: {
      variant: "primary",
      size: "md",
    },
  }
);

interface ButtonProps
  extends ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  loading?: boolean;
  icon?: React.ReactNode;
  iconRight?: React.ReactNode;
}

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      variant = "primary",
      size = "md",
      loading = false,
      icon,
      iconRight,
      className,
      children,
      disabled,
      ...props
    },
    ref
  ) => {
    return (
      <button
        ref={ref}
        className={cn(buttonVariants({ variant, size, className }))}
        disabled={disabled || loading}
        {...props}
      >
        {loading ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : icon ? (
          <span className="shrink-0">{icon}</span>
        ) : null}
        {children}
        {iconRight && !loading && (
          <span className="shrink-0">{iconRight}</span>
        )}
      </button>
    );
  }
);

Button.displayName = "Button";
export { Button, buttonVariants, type ButtonProps };
