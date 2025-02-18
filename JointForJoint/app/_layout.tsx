import { Stack } from "expo-router";

export default function RootLayout() {
  return (
    <Stack>
      {/* Ensure the app knows to use the (tabs) layout */}
      <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
    </Stack>
  );
}
