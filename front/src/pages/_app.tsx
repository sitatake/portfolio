import { AppProps } from "next/app";
import { ChakraProvider } from "@chakra-ui/react";
import { NameProvider } from "../state/nameContext";

function MyApp({ Component, pageProps }: AppProps) {
  return (
    <NameProvider>
      <ChakraProvider>
        <Component {...pageProps} />
      </ChakraProvider>
    </NameProvider>
  );
}

export default MyApp;
