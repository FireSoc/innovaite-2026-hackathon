"use client";

import { motion } from "framer-motion";
import { useWizard } from "@/hooks/useWizardStore";
import { Particles } from "@/components/magicui/Particles";
import { ShimmerButton } from "@/components/magicui/ShimmerButton";
import { Highlighter } from "@/components/ui/highlighter";
import { ArrowRight, Shield, Clock, FileCheck, Zap } from "lucide-react";
import Image from "next/image";

const trustBadges = [
  {
    icon: <Shield className="h-5 w-5" />,
    title: "FEMA Data Verified",
    desc: "Real-time disaster declaration lookup",
    gradient: "from-blue-500/10 to-blue-500/5",
    iconBg: "bg-blue-500/10 text-blue-500",
  },
  {
    icon: <Clock className="h-5 w-5" />,
    title: "30 Minutes",
    desc: "From start to submission-ready packet",
    gradient: "from-amber/10 to-amber/5",
    iconBg: "bg-amber/10 text-amber",
  },
  {
    icon: <FileCheck className="h-5 w-5" />,
    title: "Complete Bundle",
    desc: "Cover sheet, ledger, letters & evidence",
    gradient: "from-emerald-500/10 to-emerald-500/5",
    iconBg: "bg-emerald-500/10 text-emerald-500",
  },
  {
    icon: <Zap className="h-5 w-5" />,
    title: "AI-Powered",
    desc: "Smart evidence extraction & categorization",
    gradient: "from-violet-500/10 to-violet-500/5",
    iconBg: "bg-violet-500/10 text-violet-500",
  },
];

export function StepHero() {
  const { dispatch } = useWizard();

  return (
    <div className="min-h-screen flex flex-col relative overflow-hidden">
      {/* Particle background */}
      <Particles
        className="absolute inset-0 -z-10"
        quantity={80}
        staticity={30}
        ease={80}
        color="#F59E0B"
        size={0.5}
      />

      {/* Gradient orbs */}
      <div className="absolute inset-0 -z-10">
        <div className="absolute inset-0 bg-gradient-to-br from-slate-50 via-amber-50/30 to-emerald-50/20" />
        <motion.div
          className="absolute -top-32 -right-32 w-[700px] h-[700px] rounded-full opacity-[0.08]"
          style={{
            background: "radial-gradient(circle, #F59E0B 0%, transparent 70%)",
          }}
          animate={{
            scale: [1, 1.2, 1],
            x: [0, 30, 0],
            y: [0, -20, 0],
          }}
          transition={{ duration: 12, repeat: Infinity, ease: "easeInOut" }}
        />
        <motion.div
          className="absolute -bottom-32 -left-32 w-[600px] h-[600px] rounded-full opacity-[0.06]"
          style={{
            background: "radial-gradient(circle, #10B981 0%, transparent 70%)",
          }}
          animate={{
            scale: [1, 1.15, 1],
            x: [0, -20, 0],
            y: [0, 30, 0],
          }}
          transition={{ duration: 15, repeat: Infinity, ease: "easeInOut" }}
        />
        <motion.div
          className="absolute top-1/3 left-1/4 w-[400px] h-[400px] rounded-full opacity-[0.04]"
          style={{
            background: "radial-gradient(circle, #3B82F6 0%, transparent 70%)",
          }}
          animate={{
            scale: [1, 1.1, 1],
            x: [0, 15, 0],
            y: [0, -15, 0],
          }}
          transition={{ duration: 18, repeat: Infinity, ease: "easeInOut" }}
        />
        {/* Grid pattern */}
        <div
          className="absolute inset-0 opacity-[0.025]"
          style={{
            backgroundImage:
              "linear-gradient(var(--color-navy) 1px, transparent 1px), linear-gradient(to right, var(--color-navy) 1px, transparent 1px)",
            backgroundSize: "60px 60px",
          }}
        />
      </div>

      {/* Nav bar */}
      <motion.nav
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
        className="w-full px-6 py-5"
      >
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <Image src="/logo.png" alt="Remedy" width={36} height={36} className="h-9 w-9 object-contain" />
            <span className="text-lg font-bold text-foreground tracking-tight">
              Remedy
            </span>
          </div>
          <div className="hidden sm:flex items-center gap-6 text-sm text-muted-foreground">
            <span className="flex items-center gap-1.5">
              <span className="h-1.5 w-1.5 rounded-full bg-success animate-pulse" />
              Open Source
            </span>
            <span>Free Forever</span>
          </div>
        </div>
      </motion.nav>

      {/* Hero content */}
      <div className="flex-1 flex items-center justify-center px-6 pb-16">
        <div className="max-w-3xl mx-auto text-center">
          {/* Pill badge */}
          <motion.div
            initial={{ opacity: 0, y: 10, filter: "blur(4px)" }}
            animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-amber/8 border border-amber/15 text-amber-dark text-sm font-medium mb-8 backdrop-blur-sm"
          >
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-amber" />
            </span>
            Disaster Relief Made Simple
          </motion.div>

          {/* Headline */}
          <motion.h1
            initial={{ opacity: 0, y: 20, filter: "blur(8px)" }}
            animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
            transition={{ duration: 0.8, delay: 0.3, ease: [0.16, 1, 0.3, 1] }}
            className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-bold text-foreground tracking-tight leading-[1.08] mb-6"
          >
            Your relief packet,{" "}
            <Highlighter
              action="highlight"
              color="rgb(245 158 11 / 0.15)"
              strokeWidth={2}
              animationDuration={800}
              iterations={1}
              padding={4}
              isView
            >
              <span className="font-extrabold text-amber-dark">
                ready in 30 minutes
              </span>
            </Highlighter>
          </motion.h1>

          {/* Subheadline */}
          <motion.p
            initial={{ opacity: 0, y: 20, filter: "blur(4px)" }}
            animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
            transition={{ duration: 0.7, delay: 0.45, ease: [0.16, 1, 0.3, 1] }}
            className="text-lg sm:text-xl text-muted-foreground leading-relaxed max-w-2xl mx-auto mb-12"
          >
            Remedy generates a submission-ready disaster relief packet with
            FEMA eligibility verification, financial runway analysis, and
            AI-powered evidence processing — so you can focus on recovery.
          </motion.p>

          {/* CTA */}
          <motion.div
            initial={{ opacity: 0, y: 20, filter: "blur(4px)" }}
            animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
            transition={{ duration: 0.7, delay: 0.6, ease: [0.16, 1, 0.3, 1] }}
            className="flex flex-col items-center gap-4"
          >
            <ShimmerButton
              onClick={() => dispatch({ type: "NEXT_STEP" })}
              shimmerDuration="3s"
              className="shadow-[0_0_30px_rgb(245_158_11/0.2)]"
            >
              Start Your Packet
              <ArrowRight className="h-5 w-5 transition-transform group-hover:translate-x-1" />
            </ShimmerButton>
            <p className="text-sm text-muted-foreground">
              No account needed · Your data stays private · Takes ~5 minutes
            </p>
          </motion.div>
        </div>
      </div>

      {/* Trust badges */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.45, ease: [0.16, 1, 0.3, 1] }}
        className="w-full px-6 pb-12"
      >
        <div className="max-w-4xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-4">
          {trustBadges.map((badge, i) => (
            <motion.div
              key={badge.title}
              initial={{ opacity: 0, y: 16, filter: "blur(4px)" }}
              animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
              transition={{
                duration: 0.5,
                delay: 0.9 + i * 0.1,
                ease: [0.16, 1, 0.3, 1],
              }}
              className="group flex flex-col items-center text-center p-5 rounded-2xl bg-card/60 border border-border/50 backdrop-blur-sm hover:border-amber/20 hover:shadow-md hover:-translate-y-1 transition-all duration-150"
            >
              <div
                className={`h-11 w-11 rounded-xl ${badge.iconBg} flex items-center justify-center mb-3 transition-transform duration-300 group-hover:scale-110`}
              >
                {badge.icon}
              </div>
              <p className="text-sm font-semibold text-foreground mb-0.5">
                {badge.title}
              </p>
              <p className="text-xs text-muted-foreground leading-relaxed">
                {badge.desc}
              </p>
            </motion.div>
          ))}
        </div>
      </motion.div>
    </div>
  );
}
