"use client";

import { useEffect, useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";

interface Sparkle {
  id: string;
  x: string;
  y: string;
  color: string;
  delay: number;
  scale: number;
  lifespan: number;
}

interface SparklesTextProps {
  children: React.ReactNode;
  className?: string;
  sparklesCount?: number;
  colors?: { first: string; second: string };
}

export function SparklesText({
  children,
  className,
  sparklesCount = 10,
  colors = { first: "#F59E0B", second: "#FCD34D" },
}: SparklesTextProps) {
  const [sparkles, setSparkles] = useState<Sparkle[]>([]);

  const generateSparkle = useCallback((): Sparkle => {
    return {
      id: `sparkle-${Date.now()}-${Math.random()}`,
      x: `${Math.random() * 100}%`,
      y: `${Math.random() * 100}%`,
      color: Math.random() > 0.5 ? colors.first : colors.second,
      delay: Math.random() * 2,
      scale: Math.random() * 1 + 0.3,
      lifespan: Math.random() * 10 + 5,
    };
  }, [colors.first, colors.second]);

  useEffect(() => {
    const initialSparkles = Array.from({ length: sparklesCount }, () =>
      generateSparkle()
    );
    setSparkles(initialSparkles);

    const interval = setInterval(() => {
      setSparkles((current) =>
        current.map((sparkle) =>
          sparkle.lifespan <= 0 ? generateSparkle() : { ...sparkle, lifespan: sparkle.lifespan - 0.1 }
        )
      );
    }, 100);

    return () => clearInterval(interval);
  }, [sparklesCount, generateSparkle]);

  return (
    <span className={cn("relative inline-block", className)}>
      <span className="relative z-10">{children}</span>
      <AnimatePresence>
        {sparkles.map((sparkle) => (
          <motion.svg
            key={sparkle.id}
            className="pointer-events-none absolute z-20"
            initial={{ opacity: 0, scale: 0 }}
            animate={{
              opacity: [0, 1, 0],
              scale: [0, sparkle.scale, 0],
              rotate: [0, 180],
            }}
            transition={{
              duration: 1.5,
              repeat: Infinity,
              delay: sparkle.delay,
              ease: "easeInOut",
            }}
            style={{
              left: sparkle.x,
              top: sparkle.y,
              width: 16,
              height: 16,
            }}
            viewBox="0 0 160 160"
            fill="none"
          >
            <path
              d="M80 0C80 0 84.2846 41.2925 97.496 62.504C110.707 83.7155 152 #80 160 80C160 80 118.707 76.2845 97.496 97.496C76.2845 118.707 80 160 80 160C80 160 75.7154 118.707 62.504 97.496C49.2925 76.2845 0 80 0 80C0 80 41.2925 84.2846 62.504 62.504C83.7155 41.2925 80 0 80 0Z"
              fill={sparkle.color}
            />
          </motion.svg>
        ))}
      </AnimatePresence>
    </span>
  );
}
