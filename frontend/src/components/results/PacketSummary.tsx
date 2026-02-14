"use client";

import { motion } from "framer-motion";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { BorderBeam } from "@/components/magicui/BorderBeam";
import { NumberTicker } from "@/components/magicui/NumberTicker";
import {
  Download,
  FileArchive,
  FileText,
  FileSpreadsheet,
  Mail,
  CheckCircle2,
  Image as ImageIcon,
  ShieldCheck,
} from "lucide-react";

interface PacketSummaryProps {
  blob: Blob | null;
  filename: string;
  runwayDays?: number;
  businessName?: string;
  disasterId?: string;
}

const packetFiles = [
  { name: "CoverSheet.pdf", icon: <FileText className="h-4 w-4" />, desc: "Business info & disaster ID" },
  { name: "DamageSummary.pdf", icon: <FileText className="h-4 w-4" />, desc: "Bullet-point damage overview" },
  { name: "ExpenseLedger.csv", icon: <FileSpreadsheet className="h-4 w-4" />, desc: "Categorized expense data" },
  { name: "ExpenseLedger.pdf", icon: <FileText className="h-4 w-4" />, desc: "Formatted expense report" },
  { name: "EvidenceChecklist.pdf", icon: <CheckCircle2 className="h-4 w-4" />, desc: "What's included & what's missing" },
  { name: "Evidence/", icon: <ImageIcon className="h-4 w-4" />, desc: "Renamed evidence files" },
  { name: "Letters/", icon: <Mail className="h-4 w-4" />, desc: "Forbearance & waiver letters" },
];

export function PacketSummary({
  blob,
  filename,
  runwayDays,
  businessName,
  disasterId,
}: PacketSummaryProps) {
  const handleDownload = () => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div>
      <div className="flex items-center gap-3 mb-4">
        <div className="h-9 w-9 rounded-xl bg-navy/5 flex items-center justify-center">
          <ShieldCheck className="h-5 w-5 text-navy" />
        </div>
        <h2 className="text-xl font-bold text-foreground tracking-tight">
          Submission Packet
        </h2>
      </div>

      {/* Download card with BorderBeam */}
      <motion.div
        initial={{ opacity: 0, scale: 0.98 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.2, duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
        className="relative mb-6"
      >
        <Card className="bg-gradient-to-br from-navy to-navy-light text-white relative overflow-hidden border-0">
          {/* BorderBeam effect */}
          <BorderBeam
            size={250}
            duration={8}
            colorFrom="#F59E0B"
            colorTo="#10B981"
            borderWidth={2}
          />

          {/* Decorative elements */}
          <div className="absolute top-0 right-0 w-40 h-40 rounded-full bg-amber/10 -translate-y-1/2 translate-x-1/2" />
          <div className="absolute bottom-0 left-0 w-24 h-24 rounded-full bg-success/8 translate-y-1/2 -translate-x-1/2" />

          <div className="relative z-10">
            <div className="flex items-start gap-4 mb-5">
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ type: "spring", stiffness: 200, delay: 0.3 }}
                className="h-12 w-12 rounded-2xl bg-amber/20 flex items-center justify-center shrink-0 border border-amber/10"
              >
                <FileArchive className="h-6 w-6 text-amber-light" />
              </motion.div>
              <div>
                <h3 className="font-bold text-lg mb-1">{filename}</h3>
                <div className="flex flex-wrap gap-2 text-xs">
                  {businessName && (
                    <span className="text-white/60">{businessName}</span>
                  )}
                  {disasterId && (
                    <Badge variant="outline" size="sm" className="text-amber-light border-amber/30 bg-amber/10">
                      {disasterId}
                    </Badge>
                  )}
                  {runwayDays !== undefined && (
                    <span className="text-white/60">
                      <NumberTicker
                        value={Math.round(runwayDays)}
                        delay={0.5}
                        className="text-white/80 font-mono"
                      />
                      {" "}days runway
                    </span>
                  )}
                </div>
              </div>
            </div>

            <Button
              onClick={handleDownload}
              disabled={!blob}
              icon={<Download className="h-5 w-5" />}
              size="lg"
              className="w-full bg-amber hover:bg-amber-dark text-white shadow-lg shadow-amber/20"
            >
              Download ZIP Packet
            </Button>
          </div>
        </Card>
      </motion.div>

      {/* File list */}
      <Card padding="sm">
        <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3 px-2">
          Included Files
        </h3>
        <div className="space-y-0.5">
          {packetFiles.map((file, i) => (
            <motion.div
              key={file.name}
              initial={{ opacity: 0, x: 8 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4 + i * 0.06 }}
              className="flex items-center gap-3 px-3 py-2.5 rounded-xl hover:bg-accent transition-colors group"
            >
              <span className="text-muted-foreground group-hover:text-foreground transition-colors">{file.icon}</span>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-foreground">{file.name}</p>
                <p className="text-xs text-muted-foreground">{file.desc}</p>
              </div>
              <CheckCircle2 className="h-4 w-4 text-success shrink-0" />
            </motion.div>
          ))}
        </div>
      </Card>
    </div>
  );
}
