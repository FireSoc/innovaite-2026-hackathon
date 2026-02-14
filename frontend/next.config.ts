import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Proxy API calls in development to avoid CORS issues
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/:path*`,
      },
    ];
  },
};

export default nextConfig;
