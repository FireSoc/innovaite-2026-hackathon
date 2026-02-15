"use client";

import { motion } from "framer-motion";

export function Footer() {
  return (
    <footer className="w-full py-6 border-t border-border/50">
      <div className="max-w-6xl mx-auto px-4 sm:px-6">
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5 }}
          className="text-center text-sm text-muted-foreground"
        >
          <span className="font-semibold text-foreground">Remedy</span>
          <span className="mx-2">·</span>
          Disaster relief, simplified
          <span className="mx-2">·</span>
          Open source & free forever
        </motion.p>
      </div>
    </footer>
  );
}
