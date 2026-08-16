import * as React from "react";
import { cn } from "@/lib/utils";

export interface BadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: "default" | "secondary" | "outline" | "accent";
}

export function Badge({ className, variant = "default", ...props }: BadgeProps) {
  const variantStyles = {
    default: "bg-primary/90 text-primary-foreground border-cyan-300/20",
    secondary: "bg-secondary/70 text-secondary-foreground border-border/70",
    outline: "border border-border/75 text-foreground bg-transparent",
    accent: "bg-accent/18 text-cyan-100 border-cyan-300/30",
  };

  return (
    <div
      className={cn(
        "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium transition-colors select-none",
        variantStyles[variant],
        className
      )}
      {...props}
    />
  );
}
